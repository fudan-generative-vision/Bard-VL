from transformers import AutoConfig, AutoModelForImageTextToText

from nemo_automodel.components.models.bard_uni.modeling_bard_uni import BardUniForConditionalGeneration
from nemo_automodel.components.models.bard_uni.configuration_bard_uni import BardUniConfig

target_model_type = "bard_uni"
AutoConfig.register(target_model_type, BardUniConfig, exist_ok=True)
AutoModelForImageTextToText.register(BardUniConfig, BardUniForConditionalGeneration, exist_ok=True)
