import pickle
import random
import socket
import struct
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import torch


def _recvall(sock: socket.socket, num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving payload.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket):
    size_bytes = _recvall(sock, 8)
    size = struct.unpack("!Q", size_bytes)[0]
    payload = _recvall(sock, size)
    return pickle.loads(payload)


def send_message(sock: socket.socket, payload) -> None:
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(blob)))
    sock.sendall(blob)


def to_cpu_payload(data, compress_fp32_to_bf16: bool = False):
    if isinstance(data, dict):
        return {key: to_cpu_payload(value, compress_fp32_to_bf16=compress_fp32_to_bf16) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_cpu_payload(value, compress_fp32_to_bf16=compress_fp32_to_bf16) for value in data)
    if isinstance(data, torch.Tensor):
        cpu_tensor = data.detach().to("cpu")
        if compress_fp32_to_bf16 and torch.is_floating_point(cpu_tensor) and cpu_tensor.dtype in (torch.float32, torch.float64):
            return cpu_tensor.to(torch.bfloat16)
        return cpu_tensor
    return data


class TeacherRPCClient:
    def __init__(self, host: str, port: int, timeout_s: float = 600.0, compress_fp32_to_bf16: bool = False):
        self._host = host
        self._port = int(port)
        self._timeout_s = float(timeout_s)
        self._compress_fp32_to_bf16 = bool(compress_fp32_to_bf16)
        self._sock = self._connect()
        self._lock = Lock()

    def _connect(self) -> socket.socket:
        deadline = time.monotonic() + max(self._timeout_s, 1.0)
        connect_timeout_s = min(max(self._timeout_s, 1.0), 10.0)
        last_error = None
        attempt = 0

        while time.monotonic() < deadline:
            attempt += 1
            try:
                sock = socket.create_connection((self._host, self._port), timeout=connect_timeout_s)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self._timeout_s)
                return sock
            except OSError as exc:
                last_error = exc
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    break
                sleep_s = min(2.0, remaining_s, 0.2 * attempt)
                sleep_s += random.uniform(0.0, min(0.2, remaining_s))
                time.sleep(max(sleep_s, 0.05))

        if last_error is None:
            raise TimeoutError(
                f"Timed out connecting to teacher server {self._host}:{self._port} within {self._timeout_s:.1f}s."
            )
        raise TimeoutError(
            f"Timed out connecting to teacher server {self._host}:{self._port} within {self._timeout_s:.1f}s."
        ) from last_error

    def _reset_connection(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = self._connect()

    def query_topk(self, request: dict) -> dict:
        payload = to_cpu_payload(request, compress_fp32_to_bf16=self._compress_fp32_to_bf16)
        with self._lock:
            try:
                send_message(self._sock, payload)
                response = recv_message(self._sock)
            except (ConnectionError, OSError):
                self._reset_connection()
                raise

            if isinstance(response, dict) and response.get("__teacher_rpc_error__", False):
                remote_traceback = response.get("traceback", "Teacher server traceback unavailable.")
                raise RuntimeError(f"Teacher server error:\n{remote_traceback}")
            return response

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


class AsyncTeacherClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 600.0,
        max_pending: int = 2,
        compress_fp32_to_bf16: bool = False,
    ):
        self._client = TeacherRPCClient(
            host=host,
            port=port,
            timeout_s=timeout_s,
            compress_fp32_to_bf16=compress_fp32_to_bf16,
        )
        self._executor = ThreadPoolExecutor(max_workers=max(int(max_pending), 1))

    def submit(self, request: dict) -> Future:
        return self._executor.submit(self._client.query_topk, request)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        self._client.close()
