"""
Codegnipy 多提供商支持模块

支持多种 LLM 提供商：OpenAI、Anthropic、本地模型等。
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator, Iterator, TYPE_CHECKING
from enum import Enum
import json

from .streaming import StreamChunk, StreamStatus

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ProviderType(Enum):
    """提供商类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_type: ProviderType = ProviderType.OPENAI
    api_key: Optional[str] = None
    model: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    extra_params: dict = field(default_factory=dict)


class BaseProvider(ABC):
    """提供商基类"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """执行 LLM 调用"""
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[StreamChunk]:
        """执行流式调用"""
        pass
    
    @abstractmethod
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """执行异步调用"""
        pass
    
    @abstractmethod
    def stream_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """执行异步流式调用"""
        pass
    
    @abstractmethod
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        **kwargs
    ) -> Dict[str, Any]:
        """执行带工具的调用"""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI 提供商"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                raise ImportError("需要安装 openai 包。运行: pip install openai")
        return self._client
    
    def _get_async_client(self):
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                raise ImportError("需要安装 openai 包。运行: pip install openai")
        return self._async_client
    
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params
        )
        
        return response.choices[0].message.content
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
            **self.config.extra_params
        )
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated += content
                yield StreamChunk(
                    content=content,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = self._get_async_client()
        
        response = await client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params
        )
        
        return response.choices[0].message.content
    
    async def stream_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        
        response = await client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
            **self.config.extra_params
        )
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated += content
                yield StreamChunk(
                    content=content,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        **kwargs
    ) -> Dict[str, Any]:
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            tools=tools,
            tool_choice=kwargs.get("tool_choice", "auto"),
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            **self.config.extra_params
        )
        
        message = response.choices[0].message
        
        return {
            "content": message.content,
            "tool_calls": message.tool_calls,
            "message": message
        }


class AnthropicProvider(BaseProvider):
    """Anthropic 提供商"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._async_client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=self.config.api_key
                )
            except ImportError:
                raise ImportError(
                    "需要安装 anthropic 包。运行: pip install anthropic"
                )
        return self._client
    
    def _get_async_client(self):
        if self._async_client is None:
            try:
                import anthropic
                self._async_client = anthropic.AsyncAnthropic(
                    api_key=self.config.api_key
                )
            except ImportError:
                raise ImportError(
                    "需要安装 anthropic 包。运行: pip install anthropic"
                )
        return self._async_client
    
    def _convert_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> tuple[str, List[Dict[str, str]]]:
        """转换消息格式为 Anthropic 格式"""
        system = ""
        converted = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] in ("user", "assistant"):
                converted.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system, converted
    
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = self._get_client()
        system, converted = self._convert_messages(messages)
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
        }
        
        if system:
            params["system"] = system
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        params.update(self.config.extra_params)
        
        response = client.messages.create(**params)
        
        # 提取文本内容
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        
        return ""
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        system, converted = self._convert_messages(messages)
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
        }
        
        if system:
            params["system"] = system
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        params.update(self.config.extra_params)
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                accumulated += text
                yield StreamChunk(
                    content=text,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        client = self._get_async_client()
        system, converted = self._convert_messages(messages)
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
        }
        
        if system:
            params["system"] = system
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        params.update(self.config.extra_params)
        
        response = await client.messages.create(**params)
        
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text
        
        return ""
    
    async def stream_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        system, converted = self._convert_messages(messages)
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
        }
        
        if system:
            params["system"] = system
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        params.update(self.config.extra_params)
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        async with client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                accumulated += text
                yield StreamChunk(
                    content=text,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        **kwargs
    ) -> Dict[str, Any]:
        client = self._get_client()
        system, converted = self._convert_messages(messages)
        
        # 转换工具格式
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        
        params = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": converted,
            "tools": anthropic_tools,
        }
        
        if system:
            params["system"] = system
        
        params.update(self.config.extra_params)
        
        response = client.messages.create(**params)
        
        # 解析工具调用
        tool_calls = []
        content = ""
        
        for block in response.content:
            if hasattr(block, 'text'):
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return {
            "content": content,
            "tool_calls": tool_calls if tool_calls else None,
            "message": response
        }


class OllamaProvider(BaseProvider):
    """Ollama 本地模型提供商
    
    支持 Ollama 运行的本地模型，如 llama2、mistral、codellama 等。
    需要 Ollama 服务运行在本地或远程服务器上。
    
    使用示例:
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2",
            base_url="http://localhost:11434"
        )
        provider = OllamaProvider(config)
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._base_url = config.base_url or "http://localhost:11434"
    
    def _get_client(self):
        """获取 HTTP 客户端"""
        try:
            import urllib.request
            return urllib.request
        except ImportError:
            raise ImportError("需要 urllib 支持")
    
    def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        stream: bool = False
    ) -> Any:
        """发送 HTTP 请求到 Ollama"""
        import json
        import urllib.request
        
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    
    def _make_stream_request(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """发送流式 HTTP 请求"""
        import json
        import urllib.request
        
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        
        req = urllib.request.Request(
            url,
            data=json.dumps({**data, "stream": True}).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)
    
    async def _make_async_request(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> Any:
        """发送异步 HTTP 请求"""
        import aiohttp
        
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=data,
                headers={"Content-Type": "application/json"}
            ) as response:
                return await response.json()
    
    async def _make_async_stream_request(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """发送异步流式请求"""
        import json
        import aiohttp
        
        url = f"{self._base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json={**data, "stream": True},
                headers={"Content-Type": "application/json"}
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line:
                        yield json.loads(line)
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """将消息转换为 Ollama 格式的提示词"""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        prompt = self._convert_messages(messages)
        
        data = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        # 合并额外参数
        if self.config.extra_params:
            data["options"].update(self.config.extra_params)
        
        result = self._make_request("/api/generate", data)
        return result.get("response", "")
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[StreamChunk]:
        prompt = self._convert_messages(messages)
        
        data = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        if self.config.extra_params:
            data["options"].update(self.config.extra_params)
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        for chunk in self._make_stream_request("/api/generate", data):
            if "response" in chunk:
                content = chunk["response"]
                accumulated += content
                yield StreamChunk(
                    content=content,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
            
            if chunk.get("done", False):
                break
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        prompt = self._convert_messages(messages)
        
        data = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        if self.config.extra_params:
            data["options"].update(self.config.extra_params)
        
        result = await self._make_async_request("/api/generate", data)
        return result.get("response", "")
    
    async def stream_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        prompt = self._convert_messages(messages)
        
        data = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        if self.config.extra_params:
            data["options"].update(self.config.extra_params)
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        async for chunk in self._make_async_stream_request("/api/generate", data):
            if "response" in chunk:
                content = chunk["response"]
                accumulated += content
                yield StreamChunk(
                    content=content,
                    status=StreamStatus.STREAMING,
                    accumulated=accumulated
                )
            
            if chunk.get("done", False):
                break
        
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        **kwargs
    ) -> Dict[str, Any]:
        # Ollama 原生不支持工具调用，通过提示词模拟
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                desc = f"- {func['name']}: {func.get('description', 'No description')}"
                if "parameters" in func:
                    desc += f"\n  Parameters: {json.dumps(func['parameters'])}"
                tool_descriptions.append(desc)
        
        tool_prompt = "\n".join(tool_descriptions)
        enhanced_messages = messages + [{
            "role": "system",
            "content": f"\n\nAvailable tools:\n{tool_prompt}\n\nTo use a tool, respond with a JSON object."
        }]
        
        response = self.call(enhanced_messages, **kwargs)
        
        # 尝试解析 JSON 工具调用
        tool_calls = None
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                if "name" in parsed or "function" in parsed:
                    tool_calls = [{
                        "id": f"call_{hash(response) % 10000}",
                        "type": "function",
                        "function": {
                            "name": parsed.get("name") or parsed.get("function"),
                            "arguments": json.dumps(parsed.get("arguments", parsed))
                        }
                    }]
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {
            "content": response,
            "tool_calls": tool_calls,
            "message": response
        }
    
    def list_models(self) -> List[str]:
        """列出可用的本地模型"""
        import json
        import urllib.request
        
        url = f"{self._base_url.rstrip('/')}/api/tags"
        
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
                return [model["name"] for model in data.get("models", [])]
        except Exception:
            return []


class TransformersProvider(BaseProvider):
    """HuggingFace Transformers 本地模型提供商
    
    使用 transformers 库在本地运行模型。支持各种 HuggingFace 模型。
    
    使用示例:
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium",
            extra_params={"device": "cuda"}  # 或 "cpu", "mps"
        )
        provider = TransformersProvider(config)
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._model: Optional["PreTrainedModel"] = None
        self._tokenizer: Optional["PreTrainedTokenizerBase"] = None
        self._device = config.extra_params.get("device", "auto")
        self._pipeline = None
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "需要安装 transformers 和 torch。运行: pip install transformers torch"
            )
        
        model_name = self.config.model
        
        try:
            # 尝试加载 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 加载模型
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self._device if self._device != "auto" else None
            )
            
            if self._device != "auto" and hasattr(self._model, "to"):
                self._model = self._model.to(self._device)
        
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def _load_pipeline(self):
        """加载 pipeline（更简单的方式）"""
        if self._pipeline is not None:
            return
        
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "需要安装 transformers。运行: pip install transformers"
            )
        
        self._pipeline = pipeline(
            "text-generation",
            model=self.config.model,
            device=self._device if self._device != "auto" else -1
        )
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """将消息转换为提示词"""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n\n[/INST]")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(content)
        
        return "".join(prompt_parts)
    
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        self._load_model()
        
        # 类型检查：_load_model 成功后这些不为 None
        assert self._model is not None
        assert self._tokenizer is not None
        
        prompt = self._convert_messages(messages)
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        
        if self._device != "cpu" and hasattr(inputs, "to"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        max_new_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        if temperature > 0:
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )
        else:
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Iterator[StreamChunk]:
        # 本地模型流式生成需要 TextIteratorStreamer
        try:
            from transformers import TextIteratorStreamer
            import threading
        except ImportError:
            # 如果不支持流式，返回完整结果
            result = self.call(messages, **kwargs)
            yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
            yield StreamChunk(content=result, status=StreamStatus.STREAMING, accumulated=result)
            yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=result)
            return
        
        self._load_model()
        
        # 类型检查：_load_model 成功后这些不为 None
        assert self._model is not None
        assert self._tokenizer is not None
        
        prompt = self._convert_messages(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        
        if self._device != "cpu":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        max_new_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            "pad_token_id": self._tokenizer.eos_token_id
        }
        
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["do_sample"] = True
        else:
            generation_kwargs["do_sample"] = False
        
        # 在后台线程中运行生成
        thread = threading.Thread(
            target=self._model.generate,
            kwargs=generation_kwargs
        )
        thread.start()
        
        accumulated = ""
        yield StreamChunk(content="", status=StreamStatus.STARTED, accumulated="")
        
        for text in streamer:
            accumulated += text
            yield StreamChunk(
                content=text,
                status=StreamStatus.STREAMING,
                accumulated=accumulated
            )
        
        thread.join()
        yield StreamChunk(content="", status=StreamStatus.COMPLETED, accumulated=accumulated)
    
    async def call_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.call(messages, **kwargs)
        )
    
    async def stream_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        for chunk in self.stream(messages, **kwargs):
            yield chunk
            await asyncio.sleep(0)  # 让出控制权
    
    def call_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        **kwargs
    ) -> Dict[str, Any]:
        # 与 Ollama 类似，通过提示词模拟
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                desc = f"- {func['name']}: {func.get('description', 'No description')}"
                if "parameters" in func:
                    desc += f"\n  Parameters: {json.dumps(func['parameters'])}"
                tool_descriptions.append(desc)
        
        tool_prompt = "\n".join(tool_descriptions)
        enhanced_messages = messages + [{
            "role": "system",
            "content": f"\n\nAvailable tools:\n{tool_prompt}\n\nTo use a tool, respond with a JSON object."
        }]
        
        response = self.call(enhanced_messages, **kwargs)
        
        tool_calls = None
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                if "name" in parsed or "function" in parsed:
                    tool_calls = [{
                        "id": f"call_{hash(response) % 10000}",
                        "type": "function",
                        "function": {
                            "name": parsed.get("name") or parsed.get("function"),
                            "arguments": json.dumps(parsed.get("arguments", parsed))
                        }
                    }]
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {
            "content": response,
            "tool_calls": tool_calls,
            "message": response
        }


class ProviderFactory:
    """提供商工厂"""
    
    _providers: Dict[ProviderType, type] = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.HUGGINGFACE: TransformersProvider,
    }
    
    @classmethod
    def create(cls, config: ProviderConfig) -> BaseProvider:
        """创建提供商实例"""
        provider_class = cls._providers.get(config.provider_type)
        
        if provider_class is None:
            raise ValueError(f"Unknown provider type: {config.provider_type}")
        
        return provider_class(config)
    
    @classmethod
    def register(cls, provider_type: ProviderType, provider_class: type) -> None:
        """注册自定义提供商"""
        cls._providers[provider_type] = provider_class


def create_provider(
    provider_type: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """
    创建提供商实例
    
    参数:
        provider_type: 提供商类型 ("openai", "anthropic", "ollama", "huggingface", "custom")
        api_key: API 密钥 (OpenAI/Anthropic 需要)
        model: 模型名称
        base_url: API 基础 URL (Ollama 默认 http://localhost:11434)
        **kwargs: 其他配置
    
    返回:
        提供商实例
    
    示例:
        # OpenAI
        provider = create_provider("openai", api_key="sk-...", model="gpt-4")
        
        # Anthropic
        provider = create_provider("anthropic", api_key="sk-ant-...", model="claude-3-opus")
        
        # Ollama 本地模型
        provider = create_provider("ollama", model="llama2", base_url="http://localhost:11434")
        
        # HuggingFace Transformers
        provider = create_provider("huggingface", model="microsoft/DialoGPT-medium")
    """
    type_map = {
        "openai": ProviderType.OPENAI,
        "anthropic": ProviderType.ANTHROPIC,
        "ollama": ProviderType.OLLAMA,
        "huggingface": ProviderType.HUGGINGFACE,
        "custom": ProviderType.CUSTOM,
    }
    
    pt = type_map.get(provider_type.lower())
    if pt is None:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    config = ProviderConfig(
        provider_type=pt,
        api_key=api_key,
        model=model or "",
        base_url=base_url,
        **kwargs
    )
    
    return ProviderFactory.create(config)
