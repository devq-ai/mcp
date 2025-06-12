"""
AI/ML Tools Module for Agentical

This module provides comprehensive AI/ML and data processing capabilities including
language model routing, vector storage, model evaluation, batch processing,
and specialized data format processors.

Features:
- Multi-provider LLM routing and load balancing
- Vector storage and similarity search with multiple backends
- AI model evaluation and benchmarking
- Large-scale batch processing framework
- CSV parsing and analysis with data validation
- PDF processing with OCR and structure extraction
- Image analysis with computer vision capabilities
- Performance monitoring and cost optimization
- Enterprise-grade security and compliance
"""

from typing import Dict, Any, Optional

# Version information
__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "dion@devq.ai"

# Tool imports (lazy loading to handle optional dependencies)
_TOOLS_REGISTRY: Dict[str, Any] = {}

# Available tools
AVAILABLE_TOOLS = [
    "vector_store",
    "llm_router",
    "model_evaluator",
    "batch_process",
    "csv_parser",
    "pdf_processor",
    "image_analyzer"
]

def get_tool(tool_name: str, config: Optional[Dict[str, Any]] = None):
    """
    Get an AI/ML tool instance with lazy loading.

    Args:
        tool_name: Name of the tool to load
        config: Optional configuration dictionary

    Returns:
        Tool instance

    Raises:
        ImportError: If tool dependencies are not available
        ValueError: If tool name is not recognized
    """
    if tool_name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}. Available tools: {AVAILABLE_TOOLS}")

    if tool_name not in _TOOLS_REGISTRY:
        try:
            if tool_name == "vector_store":
                from .vector_store import VectorStore
                _TOOLS_REGISTRY[tool_name] = VectorStore
            elif tool_name == "llm_router":
                from .llm_router import LLMRouter
                _TOOLS_REGISTRY[tool_name] = LLMRouter
            elif tool_name == "model_evaluator":
                from .model_evaluator import ModelEvaluator
                _TOOLS_REGISTRY[tool_name] = ModelEvaluator
            elif tool_name == "batch_process":
                from .batch_process import BatchProcessor
                _TOOLS_REGISTRY[tool_name] = BatchProcessor
            elif tool_name == "csv_parser":
                from .csv_parser import CSVParser
                _TOOLS_REGISTRY[tool_name] = CSVParser
            elif tool_name == "pdf_processor":
                from .pdf_processor import PDFProcessor
                _TOOLS_REGISTRY[tool_name] = PDFProcessor
            elif tool_name == "image_analyzer":
                from .image_analyzer import ImageAnalyzer
                _TOOLS_REGISTRY[tool_name] = ImageAnalyzer
        except ImportError as e:
            raise ImportError(
                f"Cannot load {tool_name}. Missing dependencies: {e}. "
                f"Please install required packages for {tool_name}."
            )

    tool_class = _TOOLS_REGISTRY[tool_name]
    return tool_class(config or {})

def list_available_tools() -> Dict[str, str]:
    """
    List all available AI/ML tools with descriptions.

    Returns:
        Dictionary mapping tool names to descriptions
    """
    return {
        "vector_store": "Vector database for embeddings and similarity search",
        "llm_router": "Multi-provider language model routing and load balancing",
        "model_evaluator": "AI model evaluation and benchmarking framework",
        "batch_process": "Large-scale data processing and batch operations",
        "csv_parser": "Advanced CSV parsing with validation and analysis",
        "pdf_processor": "PDF text extraction, OCR, and structure analysis",
        "image_analyzer": "Computer vision and image analysis capabilities"
    }

def check_dependencies() -> Dict[str, bool]:
    """
    Check which tools have their dependencies available.

    Returns:
        Dictionary mapping tool names to availability status
    """
    status = {}

    for tool_name in AVAILABLE_TOOLS:
        try:
            # Try to import the tool to check dependencies
            get_tool(tool_name)
            status[tool_name] = True
        except ImportError:
            status[tool_name] = False

    return status

# Configuration templates for common use cases
CONFIG_TEMPLATES = {
    "development": {
        "vector_store": {
            "backend": "faiss",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "cache_enabled": True
        },
        "llm_router": {
            "providers": ["openai"],
            "default_model": "gpt-3.5-turbo",
            "max_retries": 3,
            "timeout": 30
        },
        "batch_process": {
            "batch_size": 100,
            "max_workers": 4,
            "memory_limit_mb": 1024
        }
    },
    "production": {
        "vector_store": {
            "backend": "pinecone",
            "embedding_model": "openai",
            "dimension": 1536,
            "cache_enabled": True,
            "persistence_enabled": True
        },
        "llm_router": {
            "providers": ["openai", "anthropic", "google"],
            "load_balancing": "round_robin",
            "failover_enabled": True,
            "rate_limiting": True,
            "cost_optimization": True
        },
        "batch_process": {
            "batch_size": 1000,
            "max_workers": 16,
            "memory_limit_mb": 8192,
            "persistence_enabled": True
        }
    },
    "enterprise": {
        "vector_store": {
            "backend": "weaviate",
            "embedding_model": "openai",
            "dimension": 1536,
            "encryption_enabled": True,
            "audit_logging": True,
            "high_availability": True
        },
        "llm_router": {
            "providers": ["azure_openai", "anthropic", "google", "aws_bedrock"],
            "load_balancing": "weighted_round_robin",
            "failover_enabled": True,
            "rate_limiting": True,
            "cost_optimization": True,
            "audit_logging": True,
            "compliance_mode": True
        },
        "batch_process": {
            "batch_size": 5000,
            "max_workers": 32,
            "memory_limit_mb": 16384,
            "persistence_enabled": True,
            "monitoring_enabled": True,
            "fault_tolerance": True
        }
    }
}

def get_config_template(template_name: str) -> Dict[str, Any]:
    """
    Get a configuration template for common deployment scenarios.

    Args:
        template_name: Name of the template (development, production, enterprise)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If template name is not recognized
    """
    if template_name not in CONFIG_TEMPLATES:
        available = list(CONFIG_TEMPLATES.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")

    return CONFIG_TEMPLATES[template_name].copy()

# Export main classes for direct import
__all__ = [
    "get_tool",
    "list_available_tools",
    "check_dependencies",
    "get_config_template",
    "AVAILABLE_TOOLS",
    "CONFIG_TEMPLATES"
]
