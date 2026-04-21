import os
import time
import base64
import requests
from io import BytesIO
from typing import Optional, List, Dict, Union, Any
from PIL import Image
from loguru import logger as eval_logger

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig

class VLLMProvider(ServerInterface):
    """vLLM API implementation of the Judge interface (OpenAI-Compatible)"""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        # vLLM 通常部署在本地或内网，默认 API key 可能是 "EMPTY" 或自定义
        self.api_key = os.getenv("VLLM_API_KEY", "token-vllm-judge")
        # 默认地址通常为 http://localhost:8000/v1
        self.api_url = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")

        # 初始化 OpenAI 客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
            self.use_client = True
        except ImportError:
            eval_logger.warning("OpenAI client not available, falling back to requests")
            self.use_client = False

    def is_available(self) -> bool:
        # vLLM 通常只要 URL 配置正确即可，API Key 不是硬性要求
        return bool(self.api_url)

    def evaluate(self, request: Request) -> Response:
        """Evaluate using vLLM API"""
        if not self.is_available():
            raise ValueError("vLLM API URL not configured")

        config = request.config or self.config
        messages = self.prepare_messages(request)

        # 处理多模态图像输入
        if request.images:
            messages = self._add_images_to_messages(messages, request.images)

        # 构造 Payload
        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "extra_body" :{"chat_template_kwargs": {"enable_thinking": False}}, # IMPORTANT: close thinking mode.
        }

        # vLLM 支持的额外参数
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        
        # 处理 JSON 模式 (确保vLLM 版本支持该参数)
        if config.response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        # 重试逻辑
        for attempt in range(config.num_retries):
            try:
                if self.use_client:
                    # 使用 openai SDK 调用
                    response = self.client.chat.completions.create(**payload)
                    content = response.choices[0].message.content
                    model_used = response.model
                    usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None
                    raw_response = response
                else:
                    # 使用 requests 手动调用
                    response_json = self._make_request(payload, config.timeout)
                    content = response_json["choices"][0]["message"]["content"]
                    model_used = response_json["model"]
                    usage = response_json.get("usage")
                    raw_response = response_json

                return Response(
                    content=content.strip(), 
                    model_used=model_used, 
                    usage=usage, 
                    raw_response=raw_response
                )

            except Exception as e:
                eval_logger.warning(f"vLLM Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"vLLM connection failed after {config.num_retries} attempts")
                    raise

    def _make_request(self, payload: Dict, timeout: int) -> Dict:
        """原生 HTTP 请求实现"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # 确保路径拼接正确 (vLLM 需要 /chat/completions)
        endpoint = f"{self.api_url.rstrip('/')}/chat/completions"
        
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _add_images_to_messages(self, messages: List[Dict], images: List[Union[str, bytes]]) -> List[Dict]:
        """将图像添加到最后一条 user 消息中"""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                if isinstance(messages[i]["content"], str):
                    messages[i]["content"] = [{"type": "text", "text": messages[i]["content"]}]

                for image in images:
                    base64_str = ""
                    if isinstance(image, str):
                        base64_str = self._encode_image(image)
                    elif isinstance(image, bytes):
                        # 如果已经是 bytes，假设它是 base64 编码后的文本内容
                        base64_str = image.decode("utf-8")
                    
                    messages[i]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                    })
                break
        return messages

    def _encode_image(self, image_path: str) -> str:
        """编码图像为 base64"""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")