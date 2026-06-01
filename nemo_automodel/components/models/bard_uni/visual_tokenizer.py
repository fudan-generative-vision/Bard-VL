import math

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def nearest_multiple(value, factor):
    return max(factor, int(math.floor(value / factor + 0.5)) * factor)


def resize_and_align(image: Image.Image, target_resolution: int, downsample_factor: int) -> Image.Image:
    """Resize short edge to target, keep aspect ratio, align dims to downsample_factor."""
    w, h = image.size
    if w <= h:
        new_w = target_resolution
        new_h = int(h * new_w / w)
    else:
        new_h = target_resolution
        new_w = int(w * new_h / h)

    aligned_w = nearest_multiple(new_w, downsample_factor)
    aligned_h = nearest_multiple(new_h, downsample_factor)
    return image.resize((aligned_w, aligned_h), Image.BICUBIC)


class VisualTokenizer(nn.Module):
    """Unified Visual Tokenizer interface supporting Emu3.5 IBQ and Lumina-DiMOO.

    Args:
        tokenizer_type: "ibq" (Emu3.5, codebook=131072, downsample=16) or
                        "dimoo" (Lumina-DiMOO VQ-VAE, codebook=8192, downsample=32)
        model_path: Path to the model checkpoint/directory.
        device: Target device.
    """

    TOKENIZER_CONFIGS = {
        "ibq": {"codebook_size": 131072, "downsample_factor": 16},
        "dimoo": {"codebook_size": 8192, "downsample_factor": 32},
    }

    def __init__(self, tokenizer_type: str = "ibq", model_path: str = "", device: str = "cuda"):
        super().__init__()
        if tokenizer_type not in self.TOKENIZER_CONFIGS:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}. Choose from {list(self.TOKENIZER_CONFIGS.keys())}")
        self.tokenizer_type = tokenizer_type
        self.model_path = model_path
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.tokenizer_type == "ibq":
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self._model.to(self._device).eval()
        elif self.tokenizer_type == "dimoo":
            from diffusers import VQModel
            self._model = VQModel.from_pretrained(self.model_path, subfolder="vqvae")
            self._model.to(self._device).eval()
        for p in self._model.parameters():
            p.requires_grad = False

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def codebook_size(self) -> int:
        return self.TOKENIZER_CONFIGS[self.tokenizer_type]["codebook_size"]

    @property
    def downsample_factor(self) -> int:
        return self.TOKENIZER_CONFIGS[self.tokenizer_type]["downsample_factor"]

    @torch.inference_mode()
    def encode(self, image: Image.Image, target_resolution: int = 256) -> torch.LongTensor:
        """Image → VQ codes [H/ds, W/ds] where ds = downsample_factor."""
        ds = self.downsample_factor
        image = resize_and_align(image.convert("RGB"), target_resolution, ds)
        w, h = image.size
        code_h, code_w = h // ds, w // ds

        if self.tokenizer_type == "ibq":
            return self._encode_ibq(image, code_h, code_w)
        else:
            return self._encode_dimoo(image, code_h, code_w)

    def _encode_ibq(self, image: Image.Image, code_h: int, code_w: int) -> torch.LongTensor:
        """Emu3.5 IBQ encode: normalize to [-1, 1], model.encode → indices."""
        arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
        params = next(self.model.parameters())
        tensor = tensor.to(device=params.device, dtype=params.dtype)

        _, _, info = self.model.encode(tensor)
        codes = info[-1].reshape(code_h, code_w).long()
        return codes

    def _encode_dimoo(self, image: Image.Image, code_h: int, code_w: int) -> torch.LongTensor:
        """Lumina-DiMOO VQ-VAE encode: VaeImageProcessor → encode → quantize."""
        from diffusers.image_processor import VaeImageProcessor
        vae_scale_factor = 2 ** (len(self.model.config.block_out_channels) - 1)
        processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
        x = processor.preprocess(image).to(device=self._device)
        latents = self.model.encode(x).latents
        codes = self.model.quantize(latents)[2][2].reshape(1, code_h, code_w)
        return codes[0].long()

    @torch.inference_mode()
    def decode(self, codes: torch.LongTensor) -> Image.Image:
        """VQ codes [H, W] → PIL Image."""
        if self.tokenizer_type == "ibq":
            return self._decode_ibq(codes)
        else:
            return self._decode_dimoo(codes)

    def _decode_ibq(self, codes: torch.LongTensor) -> Image.Image:
        code_h, code_w = codes.shape
        params = next(self.model.parameters())
        codes_flat = codes.to(device=params.device).flatten()

        embed_dim = self.model.quantize.embedding.embedding_dim
        recon = self.model.decode_code(
            codes_flat.unsqueeze(0),
            shape=(1, code_h, code_w, embed_dim),
        )
        img_arr = (
            (recon[0].permute(1, 2, 0).float().cpu().clamp(-1, 1) + 1) * 127.5
        ).round().clamp(0, 255).to(torch.uint8).numpy()
        return Image.fromarray(img_arr)

    def _decode_dimoo(self, codes: torch.LongTensor) -> Image.Image:
        code_h, code_w = codes.shape
        codes_flat = codes.to(device=self._device).reshape(1, code_h, code_w)
        # VQModel.decode expects quantized latents, use lookup
        embedding = self.model.quantize.embedding.weight
        quant = embedding[codes_flat.flatten()].reshape(1, code_h, code_w, -1).permute(0, 3, 1, 2)
        quant = self.model.post_quant_conv(quant)
        recon = self.model.decoder(quant)
        # recon: [1, 3, H, W] in [0, 1] (VaeImageProcessor do_normalize=False)
        img_arr = (
            recon[0].permute(1, 2, 0).float().cpu().clamp(0, 1) * 255
        ).round().to(torch.uint8).numpy()
        return Image.fromarray(img_arr)
