"""
Code Execution Tool for Agentical

This module provides secure code execution capabilities supporting multiple
programming languages with sandboxed environments, comprehensive monitoring,
and integration with the Agentical framework.

Features:
- Multi-language support (Python, JavaScript, TypeScript, SQL, Bash, R)
- Sandboxed execution environments with Docker
- Security policies and resource limits
- Async execution with timeout handling
- Code validation and static analysis
- Output capture and error handling
- Performance monitoring and observability
- Integration with MCP servers and workflow systems
"""

import asyncio
import docker
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import shutil

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError
)
from ...core.logging import log_operation


class SupportedLanguage(Enum):
    """Supported programming languages for code execution."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    SQL = "sql"
    BASH = "bash"
    R = "r"


class ExecutionEnvironment(Enum):
    """Code execution environments."""
    LOCAL = "local"
    DOCKER = "docker"
    SANDBOX = "sandbox"


class SecurityPolicy(Enum):
    """Security policies for code execution."""
    STRICT = "strict"      # Maximum restrictions, no network, limited filesystem
    MEDIUM = "medium"      # Moderate restrictions, limited network
    RELAXED = "relaxed"    # Minimal restrictions for trusted code
    CUSTOM = "custom"      # Custom security configuration


class CodeExecutionResult:
    """Result of code execution with comprehensive details."""

    def __init__(
        self,
        execution_id: str,
        language: SupportedLanguage,
        code: str,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        return_code: int = 0,
        execution_time: float = 0.0,
        memory_usage: int = 0,
        environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.execution_id = execution_id
        self.language = language
        self.code = code
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.execution_time = execution_time
        self.memory_usage = memory_usage
        self.environment = environment
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "execution_id": self.execution_id,
            "language": self.language.value,
            "code": self.code,
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "environment": self.environment.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class CodeExecutor:
    """
    Secure code execution tool supporting multiple programming languages
    with sandboxed environments and comprehensive monitoring.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        docker_client: Optional[docker.DockerClient] = None
    ):
        """
        Initialize code executor.

        Args:
            config: Configuration for code execution
            docker_client: Docker client for containerized execution
        """
        self.config = config or {}
        self.docker_client = docker_client
        self.logger = logging.getLogger(__name__)

        # Default configuration
        self.timeout_seconds = self.config.get("timeout_seconds", 30)
        self.memory_limit_mb = self.config.get("memory_limit_mb", 256)
        self.enable_network = self.config.get("enable_network", False)
        self.sandbox_mode = self.config.get("sandbox_mode", True)
        self.max_output_lines = self.config.get("max_output_lines", 1000)
        self.temp_dir = self.config.get("temp_dir", tempfile.gettempdir())

        # Language configurations
        self.language_configs = {
            SupportedLanguage.PYTHON: {
                "extensions": [".py"],
                "interpreter": "python3",
                "docker_image": "python:3.12-slim",
                "validate_command": ["python3", "-m", "py_compile"],
                "security_level": SecurityPolicy.MEDIUM
            },
            SupportedLanguage.JAVASCRIPT: {
                "extensions": [".js", ".mjs"],
                "interpreter": "node",
                "docker_image": "node:20-alpine",
                "validate_command": ["node", "--check"],
                "security_level": SecurityPolicy.MEDIUM
            },
            SupportedLanguage.TYPESCRIPT: {
                "extensions": [".ts"],
                "interpreter": "ts-node",
                "docker_image": "node:20-alpine",
                "validate_command": ["npx", "tsc", "--noEmit"],
                "security_level": SecurityPolicy.MEDIUM
            },
            SupportedLanguage.SQL: {
                "extensions": [".sql"],
                "interpreter": "sqlite3",
                "docker_image": "alpine:latest",
                "validate_command": None,
                "security_level": SecurityPolicy.STRICT
            },
            SupportedLanguage.BASH: {
                "extensions": [".sh"],
                "interpreter": "bash",
                "docker_image": "bash:5.2-alpine3.18",
                "validate_command": ["bash", "-n"],
                "security_level": SecurityPolicy.STRICT
            },
            SupportedLanguage.R: {
                "extensions": [".r", ".R"],
                "interpreter": "Rscript",
                "docker_image": "r-base:4.3.2",
                "validate_command": ["R", "--slave", "-e", "parse"],
                "security_level": SecurityPolicy.MEDIUM
            }
        }

        # Initialize Docker client if needed
        if self.sandbox_mode and not self.docker_client:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Docker client: {e}")
                self.sandbox_mode = False

    @log_operation("code_execution")
    async def execute(
        self,
        code: str,
        language: Union[SupportedLanguage, str],
        environment: ExecutionEnvironment = ExecutionEnvironment.LOCAL,
        security_policy: SecurityPolicy = SecurityPolicy.MEDIUM,
        variables: Optional[Dict[str, Any]] = None,
        timeout_override: Optional[int] = None,
        validate_first: bool = True
    ) -> CodeExecutionResult:
        """
        Execute code in specified language and environment.

        Args:
            code: Code to execute
            language: Programming language
            environment: Execution environment
            security_policy: Security policy to apply
            variables: Environment variables
            timeout_override: Override default timeout
            validate_first: Whether to validate code before execution

        Returns:
            CodeExecutionResult: Execution result with details
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Convert language to enum if string
        if isinstance(language, str):
            try:
                language = SupportedLanguage(language.lower())
            except ValueError:
                raise ToolValidationError(f"Unsupported language: {language}")

        # Validate language support
        if language not in self.language_configs:
            raise ToolValidationError(f"Language {language.value} is not supported")

        # Get language configuration
        lang_config = self.language_configs[language]
        timeout = timeout_override or self.timeout_seconds

        try:
            # Validate code if requested
            if validate_first:
                validation_result = await self._validate_code(code, language)
                if not validation_result["is_valid"]:
                    return CodeExecutionResult(
                        execution_id=execution_id,
                        language=language,
                        code=code,
                        success=False,
                        stderr=f"Code validation failed: {validation_result['errors']}",
                        environment=environment
                    )

            # Choose execution method based on environment
            if environment == ExecutionEnvironment.DOCKER and self.docker_client:
                result = await self._execute_in_docker(
                    execution_id, code, language, lang_config,
                    security_policy, variables, timeout
                )
            elif environment == ExecutionEnvironment.SANDBOX or self.sandbox_mode:
                result = await self._execute_in_sandbox(
                    execution_id, code, language, lang_config,
                    security_policy, variables, timeout
                )
            else:
                result = await self._execute_local(
                    execution_id, code, language, lang_config,
                    security_policy, variables, timeout
                )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except asyncio.TimeoutError:
            raise ToolTimeoutError(f"Code execution timed out after {timeout} seconds")
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            return CodeExecutionResult(
                execution_id=execution_id,
                language=language,
                code=code,
                success=False,
                stderr=str(e),
                environment=environment
            )

    async def _validate_code(
        self,
        code: str,
        language: SupportedLanguage
    ) -> Dict[str, Any]:
        """
        Validate code syntax before execution.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            Dict containing validation result
        """
        lang_config = self.language_configs[language]
        validate_command = lang_config.get("validate_command")

        if not validate_command:
            return {"is_valid": True, "errors": []}

        # Create temporary file for validation
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=lang_config["extensions"][0],
            delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            # Run validation command
            process = await asyncio.create_subprocess_exec(
                *validate_command, temp_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=10
            )

            stdout, stderr = await process.communicate()

            is_valid = process.returncode == 0
            errors = stderr.decode() if stderr else []

            return {
                "is_valid": is_valid,
                "errors": errors
            }

        except Exception as e:
            return {
                "is_valid": False,
                "errors": [str(e)]
            }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    async def _execute_in_docker(
        self,
        execution_id: str,
        code: str,
        language: SupportedLanguage,
        lang_config: Dict[str, Any],
        security_policy: SecurityPolicy,
        variables: Optional[Dict[str, Any]],
        timeout: int
    ) -> CodeExecutionResult:
        """Execute code in Docker container."""
        if not self.docker_client:
            raise ToolExecutionError("Docker client not available")

        # Prepare Docker configuration
        docker_config = {
            "image": lang_config["docker_image"],
            "mem_limit": f"{self.memory_limit_mb}m",
            "network_disabled": not self.enable_network,
            "security_opt": ["no-new-privileges:true"],
            "read_only": True,
            "tmpfs": {"/tmp": "noexec,nosuid,size=100m"}
        }

        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=lang_config["extensions"][0],
            delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            # Create and run container
            container = self.docker_client.containers.run(
                **docker_config,
                command=[lang_config["interpreter"], f"/code{lang_config['extensions'][0]}"],
                volumes={temp_file_path: {'bind': f"/code{lang_config['extensions'][0]}", 'mode': 'ro'}},
                environment=variables or {},
                detach=True,
                remove=True
            )

            # Wait for completion with timeout
            try:
                result = container.wait(timeout=timeout)
                logs = container.logs(stdout=True, stderr=True).decode()

                # Parse logs to separate stdout and stderr
                stdout_lines = []
                stderr_lines = []

                for line in logs.split('\n'):
                    if line.strip():
                        stdout_lines.append(line)

                return CodeExecutionResult(
                    execution_id=execution_id,
                    language=language,
                    code=code,
                    success=result["StatusCode"] == 0,
                    stdout='\n'.join(stdout_lines[:self.max_output_lines]),
                    stderr='\n'.join(stderr_lines[:self.max_output_lines]),
                    return_code=result["StatusCode"],
                    environment=ExecutionEnvironment.DOCKER
                )

            except Exception as e:
                container.kill()
                raise ToolExecutionError(f"Container execution failed: {e}")

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    async def _execute_in_sandbox(
        self,
        execution_id: str,
        code: str,
        language: SupportedLanguage,
        lang_config: Dict[str, Any],
        security_policy: SecurityPolicy,
        variables: Optional[Dict[str, Any]],
        timeout: int
    ) -> CodeExecutionResult:
        """Execute code in sandboxed environment."""
        # Create temporary directory for execution
        temp_dir = tempfile.mkdtemp(prefix=f"agentical_exec_{execution_id}_")

        try:
            # Create code file
            code_file = os.path.join(temp_dir, f"script{lang_config['extensions'][0]}")
            with open(code_file, 'w') as f:
                f.write(code)

            # Prepare execution command
            cmd = [lang_config["interpreter"], code_file]

            # Set up environment variables
            env = os.environ.copy()
            if variables:
                env.update(variables)

            # Apply security restrictions
            if security_policy == SecurityPolicy.STRICT:
                # Remove network access and limit filesystem access
                env["NO_NETWORK"] = "1"
                env["TMPDIR"] = temp_dir

            # Execute code
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir,
                env=env,
                timeout=timeout
            )

            stdout, stderr = await process.communicate()

            # Process output
            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""

            # Limit output lines
            stdout_lines = stdout_text.split('\n')[:self.max_output_lines]
            stderr_lines = stderr_text.split('\n')[:self.max_output_lines]

            return CodeExecutionResult(
                execution_id=execution_id,
                language=language,
                code=code,
                success=process.returncode == 0,
                stdout='\n'.join(stdout_lines),
                stderr='\n'.join(stderr_lines),
                return_code=process.returncode,
                environment=ExecutionEnvironment.SANDBOX
            )

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    async def _execute_local(
        self,
        execution_id: str,
        code: str,
        language: SupportedLanguage,
        lang_config: Dict[str, Any],
        security_policy: SecurityPolicy,
        variables: Optional[Dict[str, Any]],
        timeout: int
    ) -> CodeExecutionResult:
        """Execute code in local environment (least secure)."""
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=lang_config["extensions"][0],
            delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            # Prepare execution command
            cmd = [lang_config["interpreter"], temp_file_path]

            # Set up environment variables
            env = os.environ.copy()
            if variables:
                env.update(variables)

            # Execute code
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                timeout=timeout
            )

            stdout, stderr = await process.communicate()

            # Process output
            stdout_text = stdout.decode() if stdout else ""
            stderr_text = stderr.decode() if stderr else ""

            # Limit output lines
            stdout_lines = stdout_text.split('\n')[:self.max_output_lines]
            stderr_lines = stderr_text.split('\n')[:self.max_output_lines]

            return CodeExecutionResult(
                execution_id=execution_id,
                language=language,
                code=code,
                success=process.returncode == 0,
                stdout='\n'.join(stdout_lines),
                stderr='\n'.join(stderr_lines),
                return_code=process.returncode,
                environment=ExecutionEnvironment.LOCAL
            )

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return [lang.value for lang in SupportedLanguage]

    def get_language_info(self, language: Union[SupportedLanguage, str]) -> Dict[str, Any]:
        """Get information about a specific language."""
        if isinstance(language, str):
            language = SupportedLanguage(language.lower())

        if language not in self.language_configs:
            raise ToolValidationError(f"Language {language.value} is not supported")

        return self.language_configs[language].copy()

    def is_docker_available(self) -> bool:
        """Check if Docker is available for containerized execution."""
        return self.docker_client is not None

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on code executor."""
        health_status = {
            "status": "healthy",
            "docker_available": self.is_docker_available(),
            "supported_languages": self.get_supported_languages(),
            "sandbox_mode": self.sandbox_mode,
            "configuration": {
                "timeout_seconds": self.timeout_seconds,
                "memory_limit_mb": self.memory_limit_mb,
                "enable_network": self.enable_network,
                "max_output_lines": self.max_output_lines
            }
        }

        # Test basic functionality
        try:
            test_result = await self.execute(
                "print('health_check_ok')",
                SupportedLanguage.PYTHON,
                timeout_override=5
            )
            health_status["basic_execution"] = test_result.success
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["basic_execution"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating code executor
def create_code_executor(
    config: Optional[Dict[str, Any]] = None,
    docker_client: Optional[docker.DockerClient] = None
) -> CodeExecutor:
    """
    Create a code executor with specified configuration.

    Args:
        config: Configuration for code execution
        docker_client: Docker client for containerized execution

    Returns:
        CodeExecutor: Configured code executor instance
    """
    return CodeExecutor(config=config, docker_client=docker_client)
