"""
Core Development Tools Package for Agentical

This package provides essential development and execution tools that form the
foundation of the Agentical framework's development capabilities.

Core Tools Included:
- code_execution: Execute code in various programming languages
- data_analysis: Data processing, analysis, and visualization
- database_tool: Database operations and query execution
- web_search: Web search and content retrieval
- test_gen: Automatic test generation
- test_run: Test execution and reporting
- doc_gen: Documentation generation
- plan_gen: Execution plan generation
- plan_run: Plan execution and monitoring

Features:
- Multi-language code execution support (Python, JavaScript, SQL, etc.)
- Secure sandboxed execution environments
- Comprehensive data analysis with pandas, numpy, and visualization
- Database connectivity for multiple database types
- Web scraping and search capabilities
- AI-powered test and documentation generation
- Plan-based workflow execution
- Performance monitoring and observability
- Integration with MCP servers and workflow systems
"""

from .code_execution import (
    CodeExecutor,
    CodeExecutionResult,
    SupportedLanguage,
    ExecutionEnvironment,
    SecurityPolicy
)

from .data_analysis import (
    DataAnalyzer,
    AnalysisResult,
    DataSource,
    VisualizationType,
    StatisticalTest
)

from .database_tool import (
    DatabaseTool,
    DatabaseConnection,
    QueryResult,
    DatabaseType,
    TransactionContext
)

from .web_search import (
    WebSearchTool,
    SearchResult,
    SearchEngine,
    ContentType,
    SearchFilter
)

from .test_generator import (
    TestGenerator,
    GeneratedTest,
    TestFramework,
    TestType,
    CoverageReport
)

from .test_runner import (
    TestRunner,
    TestRunResult,
    TestStatus,
    TestSuite,
    TestConfiguration
)

from .doc_generator import (
    DocumentationGenerator,
    DocumentationResult,
    DocumentationType,
    OutputFormat,
    DocumentationTemplate
)

from .plan_generator import (
    PlanGenerator,
    ExecutionPlan,
    PlanStep,
    PlanType,
    PlanTemplate
)

from .plan_runner import (
    PlanRunner,
    PlanExecutionResult,
    PlanExecutionContext,
    PlanStatus,
    StepResult
)

# Core development tools
__all__ = [
    # Code execution
    "CodeExecutor",
    "CodeExecutionResult",
    "SupportedLanguage",
    "ExecutionEnvironment",
    "SecurityPolicy",

    # Data analysis
    "DataAnalyzer",
    "AnalysisResult",
    "DataSource",
    "VisualizationType",
    "StatisticalTest",

    # Database operations
    "DatabaseTool",
    "DatabaseConnection",
    "QueryResult",
    "DatabaseType",
    "TransactionContext",

    # Web search
    "WebSearchTool",
    "SearchResult",
    "SearchEngine",
    "ContentType",
    "SearchFilter",

    # Test generation
    "TestGenerator",
    "GeneratedTest",
    "TestFramework",
    "TestType",
    "CoverageReport",

    # Test execution
    "TestRunner",
    "TestRunResult",
    "TestStatus",
    "TestSuite",
    "TestConfiguration",

    # Documentation generation
    "DocumentationGenerator",
    "DocumentationResult",
    "DocumentationType",
    "OutputFormat",
    "DocumentationTemplate",

    # Plan generation
    "PlanGenerator",
    "ExecutionPlan",
    "PlanStep",
    "PlanType",
    "PlanTemplate",

    # Plan execution
    "PlanRunner",
    "PlanExecutionResult",
    "PlanExecutionContext",
    "PlanStatus",
    "StepResult"
]

# Package metadata
__version__ = "1.0.0"
__author__ = "DevQ.ai Team"
__email__ = "dion@devq.ai"

# Supported programming languages for code execution
SUPPORTED_LANGUAGES = {
    "python": {
        "extensions": [".py"],
        "interpreter": "python3",
        "docker_image": "python:3.12-slim",
        "security_level": "medium"
    },
    "javascript": {
        "extensions": [".js", ".mjs"],
        "interpreter": "node",
        "docker_image": "node:20-alpine",
        "security_level": "medium"
    },
    "typescript": {
        "extensions": [".ts"],
        "interpreter": "ts-node",
        "docker_image": "node:20-alpine",
        "security_level": "medium"
    },
    "sql": {
        "extensions": [".sql"],
        "interpreter": "sqlite3",
        "docker_image": "alpine:latest",
        "security_level": "low"
    },
    "bash": {
        "extensions": [".sh"],
        "interpreter": "bash",
        "docker_image": "bash:5.2-alpine3.18",
        "security_level": "high"
    },
    "r": {
        "extensions": [".r", ".R"],
        "interpreter": "Rscript",
        "docker_image": "r-base:4.3.2",
        "security_level": "medium"
    }
}

# Supported database types
SUPPORTED_DATABASES = {
    "sqlite": {
        "driver": "sqlite3",
        "connection_string_template": "sqlite:///{database}",
        "supports_transactions": True,
        "supports_schemas": False
    },
    "postgresql": {
        "driver": "psycopg2",
        "connection_string_template": "postgresql://{user}:{password}@{host}:{port}/{database}",
        "supports_transactions": True,
        "supports_schemas": True
    },
    "mysql": {
        "driver": "pymysql",
        "connection_string_template": "mysql+pymysql://{user}:{password}@{host}:{port}/{database}",
        "supports_transactions": True,
        "supports_schemas": True
    },
    "surrealdb": {
        "driver": "surrealdb",
        "connection_string_template": "ws://{host}:{port}/rpc",
        "supports_transactions": True,
        "supports_schemas": True
    }
}

# Supported search engines
SUPPORTED_SEARCH_ENGINES = {
    "duckduckgo": {
        "api_key_required": False,
        "rate_limit": 100,  # requests per hour
        "max_results": 20
    },
    "google": {
        "api_key_required": True,
        "rate_limit": 1000,  # requests per day
        "max_results": 100
    },
    "bing": {
        "api_key_required": True,
        "rate_limit": 3000,  # requests per month
        "max_results": 50
    },
    "serp": {
        "api_key_required": True,
        "rate_limit": 100,  # requests per month
        "max_results": 100
    }
}

# Test frameworks configuration
TEST_FRAMEWORKS = {
    "pytest": {
        "language": "python",
        "command": "pytest",
        "config_file": "pytest.ini",
        "coverage_plugin": "pytest-cov"
    },
    "jest": {
        "language": "javascript",
        "command": "npx jest",
        "config_file": "jest.config.js",
        "coverage_plugin": "built-in"
    },
    "mocha": {
        "language": "javascript",
        "command": "npx mocha",
        "config_file": "mocha.opts",
        "coverage_plugin": "nyc"
    },
    "unittest": {
        "language": "python",
        "command": "python -m unittest",
        "config_file": None,
        "coverage_plugin": "coverage"
    }
}

# Documentation formats
DOCUMENTATION_FORMATS = {
    "markdown": {
        "extension": ".md",
        "parser": "markdown",
        "supports_toc": True,
        "supports_code_blocks": True
    },
    "restructuredtext": {
        "extension": ".rst",
        "parser": "sphinx",
        "supports_toc": True,
        "supports_code_blocks": True
    },
    "html": {
        "extension": ".html",
        "parser": "html",
        "supports_toc": True,
        "supports_code_blocks": True
    },
    "pdf": {
        "extension": ".pdf",
        "parser": "pandoc",
        "supports_toc": True,
        "supports_code_blocks": True
    }
}

# Plan templates
PLAN_TEMPLATES = {
    "development": {
        "phases": ["analysis", "design", "implementation", "testing", "deployment"],
        "default_steps": [
            "analyze_requirements",
            "create_architecture",
            "implement_core_features",
            "write_tests",
            "deploy_application"
        ]
    },
    "data_analysis": {
        "phases": ["collection", "cleaning", "analysis", "visualization", "reporting"],
        "default_steps": [
            "collect_data",
            "clean_and_validate",
            "perform_analysis",
            "create_visualizations",
            "generate_report"
        ]
    },
    "testing": {
        "phases": ["planning", "unit_tests", "integration_tests", "system_tests", "reporting"],
        "default_steps": [
            "create_test_plan",
            "generate_unit_tests",
            "run_integration_tests",
            "execute_system_tests",
            "generate_test_report"
        ]
    }
}

# Default configurations
DEFAULT_CODE_EXECUTION_CONFIG = {
    "timeout_seconds": 30,
    "memory_limit_mb": 256,
    "enable_network": False,
    "sandbox_mode": True,
    "max_output_lines": 1000
}

DEFAULT_DATA_ANALYSIS_CONFIG = {
    "max_dataframe_rows": 100000,
    "max_plot_points": 10000,
    "enable_caching": True,
    "default_visualization": "matplotlib"
}

DEFAULT_DATABASE_CONFIG = {
    "connection_timeout": 30,
    "query_timeout": 300,
    "max_connections": 10,
    "enable_ssl": True
}

DEFAULT_SEARCH_CONFIG = {
    "default_engine": "duckduckgo",
    "max_results": 10,
    "timeout_seconds": 15,
    "enable_caching": True
}

DEFAULT_TEST_CONFIG = {
    "framework": "pytest",
    "coverage_threshold": 80,
    "timeout_seconds": 300,
    "parallel_execution": True
}

DEFAULT_DOCUMENTATION_CONFIG = {
    "format": "markdown",
    "include_toc": True,
    "include_examples": True,
    "template": "default"
}

DEFAULT_PLAN_CONFIG = {
    "template": "development",
    "parallel_execution": False,
    "continue_on_error": False,
    "timeout_minutes": 60
}

# Error codes for development tools
DEVELOPMENT_TOOL_ERROR_CODES = {
    "CODE_EXECUTION_FAILED": "D001",
    "LANGUAGE_NOT_SUPPORTED": "D002",
    "EXECUTION_TIMEOUT": "D003",
    "SECURITY_VIOLATION": "D004",
    "DATA_ANALYSIS_ERROR": "D005",
    "DATABASE_CONNECTION_ERROR": "D006",
    "QUERY_EXECUTION_ERROR": "D007",
    "SEARCH_ENGINE_ERROR": "D008",
    "TEST_GENERATION_ERROR": "D009",
    "TEST_EXECUTION_ERROR": "D010",
    "DOCUMENTATION_ERROR": "D011",
    "PLAN_GENERATION_ERROR": "D012",
    "PLAN_EXECUTION_ERROR": "D013"
}

def get_development_tools_info() -> dict:
    """Get comprehensive information about development tools."""
    return {
        "package_version": __version__,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "supported_databases": list(SUPPORTED_DATABASES.keys()),
        "supported_search_engines": list(SUPPORTED_SEARCH_ENGINES.keys()),
        "test_frameworks": list(TEST_FRAMEWORKS.keys()),
        "documentation_formats": list(DOCUMENTATION_FORMATS.keys()),
        "plan_templates": list(PLAN_TEMPLATES.keys()),
        "error_codes": DEVELOPMENT_TOOL_ERROR_CODES,
        "default_configs": {
            "code_execution": DEFAULT_CODE_EXECUTION_CONFIG,
            "data_analysis": DEFAULT_DATA_ANALYSIS_CONFIG,
            "database": DEFAULT_DATABASE_CONFIG,
            "search": DEFAULT_SEARCH_CONFIG,
            "testing": DEFAULT_TEST_CONFIG,
            "documentation": DEFAULT_DOCUMENTATION_CONFIG,
            "planning": DEFAULT_PLAN_CONFIG
        }
    }

def validate_development_config(tool_type: str, config: dict) -> tuple[bool, list[str]]:
    """
    Validate development tool configuration.

    Args:
        tool_type: Type of development tool
        config: Configuration dictionary

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []

    if tool_type == "code_execution":
        if "timeout_seconds" in config:
            if not isinstance(config["timeout_seconds"], int) or config["timeout_seconds"] < 1:
                errors.append("timeout_seconds must be a positive integer")

        if "memory_limit_mb" in config:
            if not isinstance(config["memory_limit_mb"], int) or config["memory_limit_mb"] < 16:
                errors.append("memory_limit_mb must be at least 16MB")

    elif tool_type == "database":
        if "connection_timeout" in config:
            if not isinstance(config["connection_timeout"], int) or config["connection_timeout"] < 1:
                errors.append("connection_timeout must be a positive integer")

    elif tool_type == "search":
        if "default_engine" in config:
            if config["default_engine"] not in SUPPORTED_SEARCH_ENGINES:
                errors.append(f"default_engine must be one of: {list(SUPPORTED_SEARCH_ENGINES.keys())}")

    elif tool_type == "testing":
        if "framework" in config:
            if config["framework"] not in TEST_FRAMEWORKS:
                errors.append(f"framework must be one of: {list(TEST_FRAMEWORKS.keys())}")

    return len(errors) == 0, errors
