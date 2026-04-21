
from transformers import AutoConfig, AutoModelForImageTextToText
from nemo_automodel.components.models.bard_vl.modeling_bard_vl import BardVLForConditionalGeneration
from nemo_automodel.components.models.bard_vl.configuration_bard_vl import BardVLConfig

target_model_type = "bard_vl"
AutoConfig.register(target_model_type, BardVLConfig, exist_ok=True)
AutoModelForImageTextToText.register(BardVLConfig, BardVLForConditionalGeneration, exist_ok=True)
