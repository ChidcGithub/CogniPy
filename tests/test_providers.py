"""
多提供商支持模块测试
"""

import pytest
from codegnipy.providers import (
    ProviderType,
    ProviderConfig,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    TransformersProvider,
    ProviderFactory,
    create_provider
)


class TestProviderType:
    """ProviderType 测试"""
    
    def test_provider_type_values(self):
        """测试提供商类型值"""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.HUGGINGFACE.value == "huggingface"
        assert ProviderType.CUSTOM.value == "custom"


class TestProviderConfig:
    """ProviderConfig 测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048
        )
        
        assert config.provider_type == ProviderType.OPENAI
        assert config.api_key == "test-key"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = ProviderConfig()
        
        assert config.provider_type == ProviderType.OPENAI
        assert config.api_key is None
        assert config.model == ""
        assert config.temperature == 0.7
        assert config.max_tokens == 1024


class TestOpenAIProvider:
    """OpenAIProvider 测试"""
    
    def test_provider_creation(self):
        """测试 OpenAI 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            model="gpt-4o-mini"
        )
        
        provider = OpenAIProvider(config)
        assert provider.config == config


class TestAnthropicProvider:
    """AnthropicProvider 测试"""
    
    def test_provider_creation(self):
        """测试 Anthropic 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            api_key="test-key",
            model="claude-3-opus"
        )
        
        provider = AnthropicProvider(config)
        assert provider.config == config
    
    def test_message_conversion(self):
        """测试消息格式转换"""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            model="claude-3-opus"
        )
        provider = AnthropicProvider(config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        system, converted = provider._convert_messages(messages)
        
        assert system == "You are helpful."
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"


class TestOllamaProvider:
    """OllamaProvider 测试"""
    
    def test_provider_creation(self):
        """测试 Ollama 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2",
            base_url="http://localhost:11434"
        )
        
        provider = OllamaProvider(config)
        assert provider.config == config
        assert provider._base_url == "http://localhost:11434"
    
    def test_default_base_url(self):
        """测试默认 base_url"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2"
        )
        
        provider = OllamaProvider(config)
        assert provider._base_url == "http://localhost:11434"
    
    def test_message_conversion(self):
        """测试消息格式转换"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2"
        )
        provider = OllamaProvider(config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        prompt = provider._convert_messages(messages)
        
        assert "System: You are helpful." in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "Assistant:" in prompt  # 结尾有助手提示
    
    def test_list_models_returns_list(self):
        """测试 list_models 返回列表（即使服务不可用）"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2",
            base_url="http://localhost:99999"  # 不存在的端口
        )
        
        provider = OllamaProvider(config)
        models = provider.list_models()
        
        # 服务不可用时返回空列表
        assert isinstance(models, list)


class TestTransformersProvider:
    """TransformersProvider 测试"""
    
    def test_provider_creation(self):
        """测试 Transformers 提供商创建"""
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium",
            extra_params={"device": "cpu"}
        )
        
        provider = TransformersProvider(config)
        assert provider.config == config
        assert provider._device == "cpu"
    
    def test_default_device(self):
        """测试默认设备"""
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium"
        )
        
        provider = TransformersProvider(config)
        assert provider._device == "auto"
    
    def test_message_conversion(self):
        """测试消息格式转换"""
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium"
        )
        provider = TransformersProvider(config)
        
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        prompt = provider._convert_messages(messages)
        
        assert "<<SYS>>" in prompt
        assert "<</SYS>>" in prompt
        assert "[INST]" in prompt


class TestProviderFactory:
    """ProviderFactory 测试"""
    
    def test_create_openai(self):
        """测试创建 OpenAI 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, OpenAIProvider)
    
    def test_create_anthropic(self):
        """测试创建 Anthropic 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            model="claude-3-opus"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, AnthropicProvider)
    
    def test_create_ollama(self):
        """测试创建 Ollama 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            model="llama2"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, OllamaProvider)
    
    def test_create_huggingface(self):
        """测试创建 HuggingFace 提供商"""
        config = ProviderConfig(
            provider_type=ProviderType.HUGGINGFACE,
            model="microsoft/DialoGPT-medium"
        )
        
        provider = ProviderFactory.create(config)
        assert isinstance(provider, TransformersProvider)


class TestCreateProvider:
    """create_provider 函数测试"""
    
    def test_create_openai_provider(self):
        """测试创建 OpenAI 提供商"""
        provider = create_provider(
            "openai",
            api_key="test-key",
            model="gpt-4"
        )
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.config.api_key == "test-key"
        assert provider.config.model == "gpt-4"
    
    def test_create_anthropic_provider(self):
        """测试创建 Anthropic 提供商"""
        provider = create_provider(
            "anthropic",
            api_key="test-key",
            model="claude-3-opus"
        )
        
        assert isinstance(provider, AnthropicProvider)
        assert provider.config.model == "claude-3-opus"
    
    def test_create_ollama_provider(self):
        """测试创建 Ollama 提供商"""
        provider = create_provider(
            "ollama",
            model="llama2",
            base_url="http://localhost:11434"
        )
        
        assert isinstance(provider, OllamaProvider)
        assert provider.config.model == "llama2"
    
    def test_create_huggingface_provider(self):
        """测试创建 HuggingFace 提供商"""
        provider = create_provider(
            "huggingface",
            model="microsoft/DialoGPT-medium"
        )
        
        assert isinstance(provider, TransformersProvider)
        assert provider.config.model == "microsoft/DialoGPT-medium"
    
    def test_invalid_provider_type(self):
        """测试无效提供商类型"""
        with pytest.raises(ValueError):
            create_provider("invalid_provider")
