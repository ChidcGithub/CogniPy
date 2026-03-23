"""
Codegnipy 外部验证模块测试
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from codegnipy.validation import (
    ExternalValidationStatus,
    Evidence,
    ExternalValidationResult,
    BaseValidator,
    WebSearchValidator,
    KnowledgeGraphValidator,
    FactCheckValidator,
    CompositeValidator,
    create_default_validator,
    verify_claim,
    verify_claim_async
)


# ============ 测试数据类 ============

class TestEvidence:
    """Evidence 测试"""
    
    def test_evidence_creation(self):
        """测试证据创建"""
        evidence = Evidence(
            source="Test Source",
            url="https://example.com",
            snippet="This is a test snippet",
            relevance=0.8,
            supports_claim=True
        )
        
        assert evidence.source == "Test Source"
        assert evidence.url == "https://example.com"
        assert evidence.snippet == "This is a test snippet"
        assert evidence.relevance == 0.8
        assert evidence.supports_claim is True
    
    def test_evidence_defaults(self):
        """测试证据默认值"""
        evidence = Evidence(source="Test")
        
        assert evidence.url is None
        assert evidence.snippet == ""
        assert evidence.relevance == 1.0
        assert evidence.supports_claim is None


class TestExternalValidationResult:
    """ExternalValidationResult 测试"""
    
    def test_result_creation(self):
        """测试结果创建"""
        result = ExternalValidationResult(
            claim="Test claim",
            status=ExternalValidationStatus.VERIFIED,
            confidence=0.9,
            summary="Test summary"
        )
        
        assert result.claim == "Test claim"
        assert result.status == ExternalValidationStatus.VERIFIED
        assert result.confidence == 0.9
        assert result.summary == "Test summary"
        assert result.evidences == []
        assert result.error is None
    
    def test_result_with_error(self):
        """测试错误结果"""
        result = ExternalValidationResult(
            claim="Test claim",
            status=ExternalValidationStatus.ERROR,
            confidence=0.0,
            error="Test error"
        )
        
        assert result.status == ExternalValidationStatus.ERROR
        assert result.error == "Test error"


# ============ 测试验证器 ============

class TestWebSearchValidator:
    """WebSearchValidator 测试"""
    
    def test_init_duckduckgo(self):
        """测试 DuckDuckGo 初始化"""
        validator = WebSearchValidator(engine="duckduckgo")
        
        assert validator.name == "web_search_duckduckgo"
        assert validator.is_available() is True
    
    def test_init_bing_without_key(self):
        """测试 Bing 无密钥初始化"""
        validator = WebSearchValidator(engine="bing")
        
        assert validator.name == "web_search_bing"
        assert validator.is_available() is False
    
    def test_init_bing_with_key(self):
        """测试 Bing 有密钥初始化"""
        validator = WebSearchValidator(engine="bing", api_key="test-key")
        
        assert validator.is_available() is True
    
    def test_extract_keywords(self):
        """测试关键词提取"""
        validator = WebSearchValidator()
        
        keywords = validator._extract_keywords("Python is a programming language")
        assert "python" in keywords
        assert "programming" in keywords
        assert "language" in keywords
        
        # 过滤停用词
        keywords = validator._extract_keywords("The quick brown fox")
        assert "the" not in keywords
        assert "quick" in keywords
    
    def test_compute_relevance(self):
        """测试相关性计算"""
        validator = WebSearchValidator()
        
        # 完全匹配
        relevance = validator._compute_relevance("python programming", "python programming")
        assert relevance == 1.0
        
        # 部分匹配 - "python code" 包含 "python"，但不含 "code"
        relevance = validator._compute_relevance("python code", "python programming")
        assert 0 < relevance <= 1
        
        # 无匹配
        relevance = validator._compute_relevance("xyz", "abc def")
        assert relevance == 0.0
    
    def test_check_support_true(self):
        """测试支持判断 - 正面"""
        validator = WebSearchValidator()
        
        supports = validator._check_support("Python exists", "This is correct and true.")
        assert supports is True
    
    def test_check_support_false(self):
        """测试支持判断 - 负面"""
        validator = WebSearchValidator()
        
        supports = validator._check_support("Python exists", "This is false and incorrect.")
        assert supports is False
    
    def test_check_support_uncertain(self):
        """测试支持判断 - 不确定"""
        validator = WebSearchValidator()
        
        supports = validator._check_support("Python exists", "Python is mentioned.")
        assert supports is None


class TestKnowledgeGraphValidator:
    """KnowledgeGraphValidator 测试"""
    
    def test_init(self):
        """测试初始化"""
        validator = KnowledgeGraphValidator()
        
        assert validator.name == "knowledge_graph_wikidata"
        assert validator.is_available() is True
    
    def test_init_with_language(self):
        """测试带语言初始化"""
        validator = KnowledgeGraphValidator(language="en")
        
        assert validator.language == "en"


class TestFactCheckValidator:
    """FactCheckValidator 测试"""
    
    def test_init_without_key(self):
        """测试无密钥初始化"""
        validator = FactCheckValidator()
        
        assert validator.name == "fact_check_google"
        assert validator.is_available() is False
    
    def test_init_with_key(self):
        """测试有密钥初始化"""
        validator = FactCheckValidator(api_key="test-key")
        
        assert validator.is_available() is True
    
    def test_parse_rating_true(self):
        """测试评级解析 - 真"""
        validator = FactCheckValidator()
        
        assert validator._parse_rating("True") is True
        assert validator._parse_rating("Correct") is True
        assert validator._parse_rating("真实") is True
    
    def test_parse_rating_false(self):
        """测试评级解析 - 假"""
        validator = FactCheckValidator()
        
        assert validator._parse_rating("False") is False
        assert validator._parse_rating("Fake") is False
        assert validator._parse_rating("谣言") is False
    
    def test_parse_rating_uncertain(self):
        """测试评级解析 - 不确定"""
        validator = FactCheckValidator()
        
        assert validator._parse_rating("Mixed") is None
        assert validator._parse_rating("Unverified") is None
        assert validator._parse_rating("Inconclusive") is None


class TestCompositeValidator:
    """CompositeValidator 测试"""
    
    def test_init_empty(self):
        """测试空初始化"""
        validator = CompositeValidator()
        
        assert validator.name == "composite_0"
        assert validator.is_available() is False
    
    def test_add_validator(self):
        """测试添加验证器"""
        validator = CompositeValidator()
        web_validator = WebSearchValidator()
        
        validator.add_validator(web_validator)
        
        assert len(validator.validators) == 1
        assert validator.is_available() is True
    
    def test_init_with_validators(self):
        """测试带验证器初始化"""
        web_validator = WebSearchValidator()
        kg_validator = KnowledgeGraphValidator()
        
        validator = CompositeValidator(validators=[web_validator, kg_validator])
        
        assert len(validator.validators) == 2
        assert validator.is_available() is True
    
    @pytest.mark.asyncio
    async def test_validate_async_no_validators(self):
        """测试无验证器时的异步验证"""
        validator = CompositeValidator()
        
        result = await validator.validate_async("Test claim")
        
        assert result.status == ExternalValidationStatus.UNAVAILABLE
        assert "没有可用的验证器" in result.error


class TestCreateDefaultValidator:
    """create_default_validator 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        validator = create_default_validator()
        
        assert validator is not None
        assert validator.is_available() is True
        assert len(validator.validators) == 2  # DuckDuckGo + Wikidata
    
    def test_with_bing_key(self):
        """测试带 Bing 密钥"""
        validator = create_default_validator(
            duckduckgo=False,
            bing_api_key="test-key"
        )
        
        assert len(validator.validators) == 2  # Bing + Wikidata
        assert any(v.name == "web_search_bing" for v in validator.validators)
    
    def test_with_fact_check_key(self):
        """测试带 Fact Check 密钥"""
        validator = create_default_validator(
            fact_check_api_key="test-key"
        )
        
        assert len(validator.validators) == 3  # DuckDuckGo + Wikidata + FactCheck
        assert any(v.name == "fact_check_google" for v in validator.validators)


class TestVerifyClaim:
    """verify_claim 测试"""
    
    def test_sync_verify(self):
        """测试同步验证"""
        # 创建 Mock 验证器
        mock_validator = MagicMock(spec=BaseValidator)
        mock_validator.is_available.return_value = True
        mock_validator.validate.return_value = ExternalValidationResult(
            claim="Test claim",
            status=ExternalValidationStatus.VERIFIED,
            confidence=0.8,
            summary="Test summary"
        )
        
        # 测试
        result = verify_claim("Test claim")
        
        assert result is not None
        assert result.claim == "Test claim"


# ============ 集成测试 ============

class TestValidatorIntegration:
    """验证器集成测试"""
    
    def test_evidence_to_result(self):
        """测试证据到结果的集成"""
        evidence = Evidence(
            source="Test Source",
            snippet="Test snippet",
            relevance=0.9,
            supports_claim=True
        )
        
        result = ExternalValidationResult(
            claim="Test claim",
            status=ExternalValidationStatus.VERIFIED,
            confidence=0.8,
            evidences=[evidence]
        )
        
        assert len(result.evidences) == 1
        assert result.evidences[0].source == "Test Source"
    
    def test_composite_validator_combines_evidences(self):
        """测试组合验证器合并证据"""
        # 使用 DuckDuckGo 和 Wikidata
        validator = CompositeValidator(validators=[
            WebSearchValidator(),
            KnowledgeGraphValidator()
        ])
        
        assert validator.is_available() is True


# ============ 运行测试 ============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
