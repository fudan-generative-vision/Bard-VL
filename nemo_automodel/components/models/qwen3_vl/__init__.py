
from transformers import AutoConfig, AutoModelForImageTextToText
from nemo_automodel.components.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from nemo_automodel.components.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

target_model_type = "qwen3_vl"
AutoConfig.register(target_model_type, Qwen3VLConfig, exist_ok=True)
AutoModelForImageTextToText.register(Qwen3VLConfig, Qwen3VLForConditionalGeneration, exist_ok=True)