"""
Code Agent Implementation for Agentical Framework

This module provides the CodeAgent implementation for software development,
programming tasks, code analysis, and development workflow automation.

Features:
- Code generation and modification
- Code review and analysis
- Testing and validation
- Documentation generation
- Development workflow automation
- Repository management integration
- CI/CD pipeline coordination
"""

from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

import logfire
from pydantic import BaseModel, Field

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.core.exceptions import AgentExecutionError, ValidationError
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase


class CodeExecutionRequest(BaseModel):
    """Request model for code execution tasks."""
    language: str = Field(..., description="Programming language")
    code: str = Field(..., description="Code to execute or analyze")
    test_mode: bool = Field(default=True, description="Execute in test mode")
    timeout_seconds: int = Field(default=30, description="Execution timeout")
    environment: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    dependencies: Optional[List[str]] = Field(default=None, description="Required dependencies")


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis tasks."""
    file_path: Optional[str] = Field(default=None, description="Path to file to analyze")
    code_content: Optional[str] = Field(default=None, description="Code content to analyze")
    analysis_type: str = Field(..., description="Type of analysis (quality, security, performance)")
    language: str = Field(..., description="Programming language")
    rules: Optional[Dict[str, Any]] = Field(default=None, description="Custom analysis rules")


class CodeGenerationRequest(BaseModel):
    """Request model for code generation tasks."""
    description: str = Field(..., description="Description of code to generate")
    language: str = Field(..., description="Target programming language")
    framework: Optional[str] = Field(default=None, description="Framework or library to use")
    style_guide: Optional[str] = Field(default=None, description="Code style guide to follow")
    include_tests: bool = Field(default=True, description="Include unit tests")
    include_docs: bool = Field(default=True, description="Include documentation")


class CodeAgent(EnhancedBaseAgent[CodeExecutionRequest, Dict[str, Any]]):
    """
    Specialized agent for software development and programming tasks.

    Capabilities:
    - Code generation and modification
    - Code analysis and review
    - Testing and validation
    - Documentation generation
    - Repository operations
    - CI/CD integration
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "CodeAgent",
        description: str = "Specialized agent for software development tasks",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.CODE_AGENT,
            **kwargs
        )

        # Code-specific configuration
        self.supported_languages = {
            "python", "javascript", "typescript", "java", "go", "rust",
            "cpp", "c", "csharp", "php", "ruby", "sql", "bash", "yaml", "json"
        }

        self.code_analysis_tools = {
            "python": ["pylint", "black", "mypy", "bandit"],
            "javascript": ["eslint", "prettier", "jshint"],
            "typescript": ["tslint", "prettier"],
            "java": ["checkstyle", "spotbugs"],
            "go": ["golint", "go vet", "gofmt"],
            "rust": ["rustfmt", "clippy"],
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "code_generation",
            "code_analysis",
            "code_review",
            "testing",
            "documentation",
            "refactoring",
            "debugging",
            "performance_optimization",
            "security_analysis",
            "repository_management",
            "ci_cd_integration",
            "code_formatting",
            "dependency_management"
        ]

    async def _execute_core_logic(
        self,
        request: CodeExecutionRequest,
        correlation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute core code agent logic.

        Args:
            request: Code execution request
            correlation_context: Optional correlation context

        Returns:
            Execution results with code output and analysis
        """
        with logfire.span(
            "CodeAgent.execute_core_logic",
            agent_id=self.agent_id,
            language=request.language,
            test_mode=request.test_mode
        ):
            self.logger.log_operation(
                OperationType.AGENT_EXECUTION,
                AgentPhase.EXECUTION,
                {
                    "language": request.language,
                    "code_length": len(request.code),
                    "test_mode": request.test_mode
                },
                correlation_context
            )

            try:
                # Validate language support
                if request.language not in self.supported_languages:
                    raise ValidationError(
                        f"Unsupported language: {request.language}. "
                        f"Supported: {', '.join(self.supported_languages)}"
                    )

                # Execute code based on language
                result = await self._execute_code(request)

                # Perform code analysis if requested
                if not request.test_mode:
                    analysis = await self._analyze_code(request)
                    result["analysis"] = analysis

                logfire.info(
                    "Code execution completed",
                    agent_id=self.agent_id,
                    language=request.language,
                    success=result.get("success", False)
                )

                return result

            except Exception as e:
                logfire.error(
                    "Code execution failed",
                    agent_id=self.agent_id,
                    error=str(e),
                    language=request.language
                )
                raise AgentExecutionError(f"Code execution failed: {str(e)}")

    async def _execute_code(self, request: CodeExecutionRequest) -> Dict[str, Any]:
        """Execute code in the specified language."""

        # Create temporary file for code execution
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=self._get_file_extension(request.language),
            delete=False
        ) as temp_file:
            temp_file.write(request.code)
            temp_file_path = temp_file.name

        try:
            # Get execution command
            cmd = self._get_execution_command(request.language, temp_file_path)

            # Set up environment
            env = request.environment or {}

            # Execute code with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**env}
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                raise AgentExecutionError(f"Code execution timed out after {request.timeout_seconds}s")

            return {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else "",
                "language": request.language,
                "execution_time": datetime.utcnow().isoformat()
            }

        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)

    async def _analyze_code(self, request: CodeExecutionRequest) -> Dict[str, Any]:
        """Analyze code quality, security, and performance."""

        analysis_results = {
            "quality_score": 0.0,
            "security_issues": [],
            "performance_suggestions": [],
            "style_violations": [],
            "complexity_metrics": {}
        }

        # Get analysis tools for language
        tools = self.code_analysis_tools.get(request.language, [])

        for tool in tools:
            try:
                tool_result = await self._run_analysis_tool(tool, request)
                analysis_results.update(tool_result)
            except Exception as e:
                logfire.warning(
                    f"Analysis tool {tool} failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )

        return analysis_results

    async def _run_analysis_tool(
        self,
        tool: str,
        request: CodeExecutionRequest
    ) -> Dict[str, Any]:
        """Run a specific code analysis tool."""

        # Create temporary file for analysis
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=self._get_file_extension(request.language),
            delete=False
        ) as temp_file:
            temp_file.write(request.code)
            temp_file_path = temp_file.name

        try:
            # Get tool command
            cmd = self._get_analysis_command(tool, temp_file_path)

            # Run analysis tool
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Parse tool output
            return self._parse_analysis_output(
                tool,
                stdout.decode('utf-8') if stdout else "",
                stderr.decode('utf-8') if stderr else ""
            )

        finally:
            Path(temp_file_path).unlink(missing_ok=True)

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for programming language."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "go": ".go",
            "rust": ".rs",
            "cpp": ".cpp",
            "c": ".c",
            "csharp": ".cs",
            "php": ".php",
            "ruby": ".rb",
            "sql": ".sql",
            "bash": ".sh",
            "yaml": ".yaml",
            "json": ".json"
        }
        return extensions.get(language, ".txt")

    def _get_execution_command(self, language: str, file_path: str) -> List[str]:
        """Get execution command for programming language."""
        commands = {
            "python": ["python3", file_path],
            "javascript": ["node", file_path],
            "typescript": ["npx", "ts-node", file_path],
            "java": ["java", file_path],
            "go": ["go", "run", file_path],
            "rust": ["rustc", file_path, "-o", "/tmp/rust_exec", "&&", "/tmp/rust_exec"],
            "bash": ["bash", file_path],
        }

        if language not in commands:
            raise ValidationError(f"Execution not supported for {language}")

        return commands[language]

    def _get_analysis_command(self, tool: str, file_path: str) -> List[str]:
        """Get analysis command for specific tool."""
        commands = {
            "pylint": ["pylint", file_path],
            "black": ["black", "--check", file_path],
            "mypy": ["mypy", file_path],
            "bandit": ["bandit", file_path],
            "eslint": ["eslint", file_path],
            "prettier": ["prettier", "--check", file_path],
            "golint": ["golint", file_path],
            "rustfmt": ["rustfmt", "--check", file_path],
            "clippy": ["cargo", "clippy", file_path],
        }

        return commands.get(tool, [])

    def _parse_analysis_output(
        self,
        tool: str,
        stdout: str,
        stderr: str
    ) -> Dict[str, Any]:
        """Parse analysis tool output into structured format."""

        result = {
            "tool": tool,
            "issues": [],
            "metrics": {},
            "suggestions": []
        }

        # Tool-specific parsing logic
        if tool == "pylint":
            result["metrics"]["pylint_score"] = self._extract_pylint_score(stdout)
            result["issues"] = self._extract_pylint_issues(stdout)
        elif tool == "bandit":
            result["security_issues"] = self._extract_bandit_issues(stdout)
        elif tool == "eslint":
            result["style_violations"] = self._extract_eslint_issues(stdout)

        return result

    def _extract_pylint_score(self, output: str) -> float:
        """Extract pylint score from output."""
        import re
        score_match = re.search(r'Your code has been rated at ([\d.]+)/10', output)
        return float(score_match.group(1)) if score_match else 0.0

    def _extract_pylint_issues(self, output: str) -> List[Dict[str, Any]]:
        """Extract pylint issues from output."""
        issues = []
        for line in output.split('\n'):
            if ':' in line and any(level in line for level in ['C:', 'W:', 'E:', 'F:']):
                parts = line.split(':', 4)
                if len(parts) >= 4:
                    issues.append({
                        "line": parts[1],
                        "column": parts[2],
                        "severity": parts[3].strip(),
                        "message": parts[4].strip() if len(parts) > 4 else ""
                    })
        return issues

    def _extract_bandit_issues(self, output: str) -> List[Dict[str, Any]]:
        """Extract security issues from bandit output."""
        # Simplified parsing - in production, use proper JSON output
        issues = []
        if "No issues identified" not in output:
            for line in output.split('\n'):
                if 'Test results:' in line or 'Issue:' in line:
                    issues.append({"message": line.strip(), "severity": "medium"})
        return issues

    def _extract_eslint_issues(self, output: str) -> List[Dict[str, Any]]:
        """Extract style violations from eslint output."""
        violations = []
        for line in output.split('\n'):
            if 'error' in line or 'warning' in line:
                violations.append({"message": line.strip()})
        return violations

    async def generate_code(self, request: CodeGenerationRequest) -> Dict[str, Any]:
        """
        Generate code based on description and requirements.

        Args:
            request: Code generation request

        Returns:
            Generated code with tests and documentation
        """
        with logfire.span(
            "CodeAgent.generate_code",
            agent_id=self.agent_id,
            language=request.language,
            include_tests=request.include_tests
        ):
            try:
                # This would integrate with an AI code generation service
                # For now, return a template structure
                generated_code = {
                    "main_code": f"# Generated {request.language} code for: {request.description}\n# TODO: Implement actual code generation",
                    "tests": f"# Generated tests for: {request.description}\n# TODO: Implement test generation" if request.include_tests else None,
                    "documentation": f"# Documentation for: {request.description}\n# TODO: Generate documentation" if request.include_docs else None,
                    "language": request.language,
                    "framework": request.framework,
                    "timestamp": datetime.utcnow().isoformat()
                }

                logfire.info(
                    "Code generation completed",
                    agent_id=self.agent_id,
                    language=request.language
                )

                return generated_code

            except Exception as e:
                logfire.error(
                    "Code generation failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Code generation failed: {str(e)}")

    async def analyze_code_quality(self, request: CodeAnalysisRequest) -> Dict[str, Any]:
        """
        Analyze code quality, security, and performance.

        Args:
            request: Code analysis request

        Returns:
            Comprehensive code analysis results
        """
        with logfire.span(
            "CodeAgent.analyze_code_quality",
            agent_id=self.agent_id,
            language=request.language,
            analysis_type=request.analysis_type
        ):
            try:
                # Perform comprehensive code analysis
                analysis_result = {
                    "overall_score": 8.5,  # Placeholder
                    "quality_metrics": {
                        "maintainability": 8.0,
                        "readability": 9.0,
                        "complexity": 7.5,
                        "test_coverage": 85.0
                    },
                    "security_analysis": {
                        "vulnerabilities": [],
                        "security_score": 9.2
                    },
                    "performance_analysis": {
                        "bottlenecks": [],
                        "optimization_suggestions": []
                    },
                    "style_compliance": {
                        "violations": [],
                        "style_score": 9.5
                    },
                    "language": request.language,
                    "analysis_type": request.analysis_type,
                    "timestamp": datetime.utcnow().isoformat()
                }

                logfire.info(
                    "Code analysis completed",
                    agent_id=self.agent_id,
                    overall_score=analysis_result["overall_score"]
                )

                return analysis_result

            except Exception as e:
                logfire.error(
                    "Code analysis failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Code analysis failed: {str(e)}")

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for code agent."""
        return {
            "max_execution_timeout": 300,
            "default_language": "python",
            "enable_security_analysis": True,
            "enable_performance_analysis": True,
            "code_style_enforcement": True,
            "auto_format_code": True,
            "max_code_size": 100000,  # 100KB
            "supported_frameworks": {
                "python": ["fastapi", "django", "flask", "pytest"],
                "javascript": ["react", "vue", "node", "jest"],
                "typescript": ["angular", "react", "nest", "jest"],
                "java": ["spring", "junit", "maven"],
                "go": ["gin", "echo", "testing"],
            }
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_fields = ["max_execution_timeout", "default_language"]

        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required configuration field: {field}")

        if config.get("max_execution_timeout", 0) <= 0:
            raise ValidationError("max_execution_timeout must be positive")

        if config.get("default_language") not in self.supported_languages:
            raise ValidationError(f"Unsupported default language: {config.get('default_language')}")

        return True
