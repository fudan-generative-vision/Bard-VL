from nemo_automodel.components.models.bard_vl.configuration_bard_vl import (
    BardVLConfig,
    BardVLTextConfig,
    BardVLVisionConfig,
)
from nemo_automodel.components.models.utils import register_with_transformers_autodoc


class BardUniConfig(BardVLConfig):
    model_type = "bard_uni"
    sub_configs = {"vision_config": BardVLVisionConfig, "text_config": BardVLTextConfig}

    def __init__(
        self,
        # VQ Tokenizer
        vq_tokenizer_type: str = "ibq",
        vq_model_path: str = "",
        vq_codebook_size: int = 131072,
        vq_downsample_factor: int = 16,
        vq_weight_tying: bool = False,
        # Special token IDs (reuse Qwen vocab placeholders)
        img_gen_start_token_id: int = 151672,
        img_gen_end_token_id: int = 151673,
        img_newline_token_id: int = 151674,
        img_src_start_token_id: int = 151675,
        img_src_end_token_id: int = 151676,
        uncondition_token_id: int = 151677,
        # Newline ablation
        use_newline: bool = False,
        mask_newline: bool = False,
        # Generation config
        max_image_tokens: int = 1024,
        image_gen_timesteps: int = 64,
        cfg_scale: float = 4.0,
        cfg_uncond_drop_rate: float = 0.1,
        **kwargs,
    ):
        self.vq_tokenizer_type = vq_tokenizer_type
        self.vq_model_path = vq_model_path
        self.vq_codebook_size = vq_codebook_size
        self.vq_downsample_factor = vq_downsample_factor
        self.vq_weight_tying = vq_weight_tying

        self.img_gen_start_token_id = img_gen_start_token_id
        self.img_gen_end_token_id = img_gen_end_token_id
        self.img_newline_token_id = img_newline_token_id
        self.img_src_start_token_id = img_src_start_token_id
        self.img_src_end_token_id = img_src_end_token_id
        self.uncondition_token_id = uncondition_token_id

        self.use_newline = use_newline
        self.mask_newline = mask_newline

        self.max_image_tokens = max_image_tokens
        self.image_gen_timesteps = image_gen_timesteps
        self.cfg_scale = cfg_scale
        self.cfg_uncond_drop_rate = cfg_uncond_drop_rate

        super().__init__(**kwargs)


register_with_transformers_autodoc(BardUniConfig.model_type, BardUniConfig)

__all__ = ["BardUniConfig"]
