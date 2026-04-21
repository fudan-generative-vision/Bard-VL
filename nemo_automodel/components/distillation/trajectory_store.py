import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import torch


class DiskTrajectoryStore:
    def __init__(self, root_dir: str | os.PathLike[str], keep_consumed: bool = False):
        self.root_dir = Path(root_dir)
        self.keep_consumed = bool(keep_consumed)
        self.ready_dir = self.root_dir / "ready"
        self.claimed_dir = self.root_dir / "claimed"
        self.consumed_dir = self.root_dir / "consumed"
        self.producers_dir = self.root_dir / "producers"
        for directory in (self.root_dir, self.ready_dir, self.claimed_dir, self.consumed_dir, self.producers_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _version_dirname(self, version: str | None) -> str:
        if version is None:
            return "unknown"
        return str(version).replace(os.sep, "_")

    def _ready_version_dir(self, version: str | None) -> Path:
        version_dir = self.ready_dir / self._version_dirname(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        return version_dir

    def count_ready(self, version: str | None = None) -> int:
        if version is None:
            return sum(1 for _ in self.ready_dir.glob("*/*.pt"))
        return sum(1 for _ in self._ready_version_dir(version).glob("*.pt"))

    def list_ready_versions(self) -> list[str]:
        version_dirs = [path for path in self.ready_dir.iterdir() if path.is_dir()]
        return sorted(path.name for path in version_dirs if any(path.glob("*.pt")))

    def put(self, record: dict[str, Any], prefix: str = "traj") -> Path:
        record_id = record.get("record_id", f"{int(time.time() * 1e6)}-{uuid.uuid4().hex}")
        filename = f"{prefix}-{record_id}.pt"
        tmp_path = self.root_dir / f".tmp-{filename}"
        ready_path = self._ready_version_dir(record.get("model_version", "unknown")) / filename
        torch.save(record, tmp_path)
        os.replace(tmp_path, ready_path)
        return ready_path

    def claim_next(
        self,
        timeout_s: float | None = None,
        poll_interval_s: float = 1.0,
        consumer_id: str | None = None,
        version: str | None = None,
    ) -> tuple[Path | None, dict[str, Any] | None]:
        if consumer_id is None:
            consumer_id = f"{socket.gethostname()}-{os.getpid()}"
        deadline = None if timeout_s is None else time.time() + float(timeout_s)
        while True:
            if version is None:
                ready_paths = sorted(self.ready_dir.glob("*/*.pt"))
            else:
                ready_paths = sorted(self._ready_version_dir(version).glob("*.pt"))
            for ready_path in ready_paths:
                claimed_name = f"{ready_path.stem}.{consumer_id}.claimed{ready_path.suffix}"
                claimed_path = self.claimed_dir / claimed_name
                try:
                    os.replace(ready_path, claimed_path)
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
                payload = torch.load(claimed_path, map_location="cpu")
                return claimed_path, payload
            if deadline is not None and time.time() >= deadline:
                return None, None
            time.sleep(max(float(poll_interval_s), 0.01))

    def ack(self, claimed_path: str | os.PathLike[str]) -> None:
        path = Path(claimed_path)
        if not path.exists():
            return
        if self.keep_consumed:
            consumed_path = self.consumed_dir / path.name.replace(".claimed", ".done")
            os.replace(path, consumed_path)
            return
        path.unlink()

    def requeue(self, claimed_path: str | os.PathLike[str]) -> Path:
        path = Path(claimed_path)
        if not path.exists():
            raise FileNotFoundError(f"Claimed path does not exist: {path}")
        ready_name = path.name.replace(".claimed", "")
        ready_path = self.ready_dir / ready_name
        os.replace(path, ready_path)
        return ready_path

    def mark_producer_done(self, producer_id: str, metadata: dict[str, Any] | None = None) -> Path:
        tmp_path = self.root_dir / f".tmp-producer-{producer_id}.json"
        done_path = self.producers_dir / f"{producer_id}.done.json"
        payload = {
            "producer_id": producer_id,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, done_path)
        return done_path

    def list_done_producers(self) -> list[str]:
        producer_files = sorted(self.producers_dir.glob("*.done.json"))
        return [path.name[: -len(".done.json")] for path in producer_files]

    def all_expected_producers_done(self, expected_count: int) -> bool:
        if expected_count <= 0:
            return False
        return len(self.list_done_producers()) >= int(expected_count)


class VersionedModelRegistry:
    def __init__(self, root_dir: str | os.PathLike[str]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir = self.root_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.root_dir / "latest.json"

    def version_dir(self, version: str) -> Path:
        return self.versions_dir / str(version)

    def publish_state_dict(
        self,
        state_dict: dict[str, Any],
        version: str,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version = str(version)
        version_dir = self.version_dir(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        state_path = version_dir / "state.pt"
        metadata_path = version_dir / "metadata.json"
        latest_tmp = self.root_dir / ".latest.json.tmp"

        cpu_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.detach().cpu()
            else:
                cpu_state_dict[key] = value
        torch.save(cpu_state_dict, state_path)

        payload = {
            "version": version,
            "step": int(step),
            "state_path": str(state_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        latest_tmp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")
        os.replace(latest_tmp, self.latest_path)
        return payload

    def read_latest(self) -> dict[str, Any] | None:
        if not self.latest_path.exists():
            return None
        return json.loads(self.latest_path.read_text(encoding="utf-8"))

    def load_latest_state_dict(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        latest = self.read_latest()
        if latest is None:
            return None, None
        state_dict = torch.load(latest["state_path"], map_location="cpu")
        return state_dict, latest

    def load_state_dict_for_version(self, version: str) -> tuple[dict[str, Any], dict[str, Any]]:
        version_dir = self.version_dir(str(version))
        metadata_path = version_dir / "metadata.json"
        state_path = version_dir / "state.pt"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        state_dict = torch.load(state_path, map_location="cpu")
        return state_dict, metadata
