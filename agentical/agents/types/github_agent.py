"""
GitHub Agent Implementation for Agentical Framework

This module provides the GitHubAgent implementation for GitHub operations,
repository management, code review automation, and GitHub workflow orchestration.

Features:
- Repository management and operations
- Pull request automation and review
- Issue tracking and management
- GitHub Actions workflow automation
- Code analysis and quality checks
- Branch management and protection
- Release management and deployment
- Team and access management
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime
import asyncio
import json
import base64
from pathlib import Path

import logfire
from pydantic import BaseModel, Field

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.core.exceptions import AgentExecutionError, ValidationError
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase


class RepositoryRequest(BaseModel):
    """Request model for repository operations."""
    repository: str = Field(..., description="Repository name (owner/repo)")
    action: str = Field(..., description="Action to perform")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Action parameters")
    branch: Optional[str] = Field(default="main", description="Target branch")
    path: Optional[str] = Field(default=None, description="File/directory path")


class PullRequestRequest(BaseModel):
    """Request model for pull request operations."""
    repository: str = Field(..., description="Repository name (owner/repo)")
    action: str = Field(..., description="PR action (create, review, merge, close)")
    title: Optional[str] = Field(default=None, description="PR title")
    body: Optional[str] = Field(default=None, description="PR description")
    head_branch: Optional[str] = Field(default=None, description="Source branch")
    base_branch: str = Field(default="main", description="Target branch")
    pr_number: Optional[int] = Field(default=None, description="PR number for existing PR")
    review_action: Optional[str] = Field(default=None, description="Review action (approve, request_changes, comment)")
    review_body: Optional[str] = Field(default=None, description="Review comment")


class IssueRequest(BaseModel):
    """Request model for issue management."""
    repository: str = Field(..., description="Repository name (owner/repo)")
    action: str = Field(..., description="Issue action (create, update, close, assign)")
    title: Optional[str] = Field(default=None, description="Issue title")
    body: Optional[str] = Field(default=None, description="Issue description")
    issue_number: Optional[int] = Field(default=None, description="Issue number for existing issue")
    labels: Optional[List[str]] = Field(default=None, description="Issue labels")
    assignees: Optional[List[str]] = Field(default=None, description="Issue assignees")
    milestone: Optional[str] = Field(default=None, description="Issue milestone")


class WorkflowRequest(BaseModel):
    """Request model for GitHub Actions workflows."""
    repository: str = Field(..., description="Repository name (owner/repo)")
    action: str = Field(..., description="Workflow action (create, run, cancel, status)")
    workflow_name: Optional[str] = Field(default=None, description="Workflow name")
    workflow_file: Optional[str] = Field(default=None, description="Workflow file path")
    run_id: Optional[int] = Field(default=None, description="Workflow run ID")
    ref: str = Field(default="main", description="Git reference to run workflow on")
    inputs: Optional[Dict[str, Any]] = Field(default=None, description="Workflow inputs")


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis operations."""
    repository: str = Field(..., description="Repository name (owner/repo)")
    analysis_type: str = Field(..., description="Type of analysis (security, quality, dependencies)")
    path: Optional[str] = Field(default=None, description="Specific path to analyze")
    language: Optional[str] = Field(default=None, description="Programming language")
    ref: str = Field(default="main", description="Git reference to analyze")


class GitHubAgent(EnhancedBaseAgent[RepositoryRequest, Dict[str, Any]]):
    """
    Specialized agent for GitHub operations and repository management.

    Capabilities:
    - Repository management and operations
    - Pull request automation and review
    - Issue tracking and management
    - GitHub Actions workflow automation
    - Code analysis and security scanning
    - Branch management and protection
    - Release management
    - Team and access management
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "GitHubAgent",
        description: str = "Specialized agent for GitHub operations and repository management",
        github_token: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.GITHUB_AGENT,
            **kwargs
        )

        self.github_token = github_token
        self.api_base_url = "https://api.github.com"

        # GitHub-specific configuration
        self.supported_actions = {
            "repository": [
                "create", "delete", "get", "update", "list", "fork", "clone",
                "get_contents", "create_file", "update_file", "delete_file",
                "get_branches", "create_branch", "delete_branch", "protect_branch"
            ],
            "pull_request": [
                "create", "get", "list", "update", "merge", "close",
                "review", "request_review", "get_reviews", "get_files"
            ],
            "issue": [
                "create", "get", "list", "update", "close", "reopen",
                "assign", "unassign", "add_labels", "remove_labels"
            ],
            "workflow": [
                "create", "get", "list", "run", "cancel", "get_runs",
                "get_run", "download_logs", "get_artifacts"
            ],
            "release": [
                "create", "get", "list", "update", "delete", "upload_asset"
            ]
        }

        self.code_analysis_tools = {
            "security": ["codeql", "dependabot", "secret_scanning"],
            "quality": ["sonarqube", "codeclimate", "lgtm"],
            "dependencies": ["dependabot", "snyk", "whitesource"]
        }

        self.webhook_events = [
            "push", "pull_request", "issues", "release", "workflow_run",
            "check_suite", "deployment", "fork", "star", "watch"
        ]

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "repository_management",
            "pull_request_automation",
            "issue_management",
            "code_review_automation",
            "workflow_automation",
            "branch_management",
            "release_management",
            "security_scanning",
            "dependency_management",
            "team_management",
            "access_control",
            "webhook_management",
            "api_integration",
            "git_operations",
            "continuous_integration",
            "deployment_automation",
            "project_management",
            "collaboration_tools",
            "documentation_automation",
            "compliance_checking"
        ]

    async def _execute_core_logic(
        self,
        request: RepositoryRequest,
        correlation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute core GitHub agent logic.

        Args:
            request: Repository operation request
            correlation_context: Optional correlation context

        Returns:
            Operation results with GitHub API responses
        """
        with logfire.span(
            "GitHubAgent.execute_core_logic",
            agent_id=self.agent_id,
            repository=request.repository,
            action=request.action
        ):
            self.logger.log_operation(
                OperationType.AGENT_EXECUTION,
                AgentPhase.EXECUTION,
                {
                    "repository": request.repository,
                    "action": request.action,
                    "branch": request.branch
                },
                correlation_context
            )

            try:
                # Validate action support
                if request.action not in self.supported_actions.get("repository", []):
                    raise ValidationError(f"Unsupported repository action: {request.action}")

                # Execute action
                if request.action == "get":
                    result = await self._get_repository(request)
                elif request.action == "create":
                    result = await self._create_repository(request)
                elif request.action == "update":
                    result = await self._update_repository(request)
                elif request.action == "get_contents":
                    result = await self._get_repository_contents(request)
                elif request.action == "create_file":
                    result = await self._create_file(request)
                elif request.action == "update_file":
                    result = await self._update_file(request)
                elif request.action == "get_branches":
                    result = await self._get_branches(request)
                elif request.action == "create_branch":
                    result = await self._create_branch(request)
                elif request.action == "protect_branch":
                    result = await self._protect_branch(request)
                else:
                    result = await self._execute_generic_action(request)

                # Add metadata
                result.update({
                    "repository": request.repository,
                    "action": request.action,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id
                })

                logfire.info(
                    "GitHub operation completed",
                    agent_id=self.agent_id,
                    repository=request.repository,
                    action=request.action,
                    success=result.get("success", False)
                )

                return result

            except Exception as e:
                logfire.error(
                    "GitHub operation failed",
                    agent_id=self.agent_id,
                    error=str(e),
                    repository=request.repository,
                    action=request.action
                )
                raise AgentExecutionError(f"GitHub operation failed: {str(e)}")

    async def _make_github_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make authenticated GitHub API request."""

        # Mock GitHub API response for demonstration
        # In production, this would use aiohttp or httpx to make actual API calls
        mock_responses = {
            "GET /repos": {
                "id": 123456789,
                "name": "test-repo",
                "full_name": "user/test-repo",
                "private": False,
                "default_branch": "main",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": datetime.utcnow().isoformat(),
                "clone_url": "https://github.com/user/test-repo.git"
            },
            "POST /repos": {
                "id": 123456790,
                "name": "new-repo",
                "full_name": "user/new-repo",
                "private": False,
                "default_branch": "main",
                "created_at": datetime.utcnow().isoformat()
            },
            "GET /contents": {
                "name": "README.md",
                "path": "README.md",
                "type": "file",
                "size": 1024,
                "download_url": "https://github.com/user/repo/raw/main/README.md",
                "content": base64.b64encode(b"# Test Repository\n\nThis is a test.").decode()
            },
            "GET /branches": [
                {
                    "name": "main",
                    "protected": True,
                    "commit": {
                        "sha": "abc123def456",
                        "url": "https://api.github.com/repos/user/repo/commits/abc123def456"
                    }
                },
                {
                    "name": "develop",
                    "protected": False,
                    "commit": {
                        "sha": "def456abc123",
                        "url": "https://api.github.com/repos/user/repo/commits/def456abc123"
                    }
                }
            ]
        }

        # Return appropriate mock response
        key = f"{method} /{endpoint.split('/')[-1]}"
        return mock_responses.get(key, {"success": True, "data": {}})

    async def _get_repository(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Get repository information."""
        response = await self._make_github_request(
            "GET",
            f"repos/{request.repository}"
        )

        return {
            "success": True,
            "repository_info": response,
            "operation": "get_repository"
        }

    async def _create_repository(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Create a new repository."""
        params = request.parameters or {}

        data = {
            "name": params.get("name", request.repository.split("/")[-1]),
            "description": params.get("description", ""),
            "private": params.get("private", False),
            "auto_init": params.get("auto_init", True),
            "gitignore_template": params.get("gitignore_template"),
            "license_template": params.get("license_template")
        }

        response = await self._make_github_request(
            "POST",
            "user/repos",
            data=data
        )

        return {
            "success": True,
            "repository_info": response,
            "operation": "create_repository"
        }

    async def _update_repository(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Update repository settings."""
        params = request.parameters or {}

        data = {
            "name": params.get("name"),
            "description": params.get("description"),
            "private": params.get("private"),
            "has_issues": params.get("has_issues"),
            "has_projects": params.get("has_projects"),
            "has_wiki": params.get("has_wiki"),
            "default_branch": params.get("default_branch")
        }

        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}

        response = await self._make_github_request(
            "PATCH",
            f"repos/{request.repository}",
            data=data
        )

        return {
            "success": True,
            "repository_info": response,
            "operation": "update_repository"
        }

    async def _get_repository_contents(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Get repository contents."""
        params = {
            "path": request.path or "",
            "ref": request.branch
        }

        response = await self._make_github_request(
            "GET",
            f"repos/{request.repository}/contents/{request.path or ''}",
            params=params
        )

        return {
            "success": True,
            "contents": response,
            "path": request.path,
            "operation": "get_contents"
        }

    async def _create_file(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Create a file in the repository."""
        params = request.parameters or {}

        data = {
            "message": params.get("commit_message", "Create file via Agentical"),
            "content": base64.b64encode(params.get("content", "").encode()).decode(),
            "branch": request.branch
        }

        response = await self._make_github_request(
            "PUT",
            f"repos/{request.repository}/contents/{request.path}",
            data=data
        )

        return {
            "success": True,
            "file_info": response,
            "path": request.path,
            "operation": "create_file"
        }

    async def _update_file(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Update an existing file in the repository."""
        params = request.parameters or {}

        # First get the current file to obtain its SHA
        current_file = await self._make_github_request(
            "GET",
            f"repos/{request.repository}/contents/{request.path}",
            params={"ref": request.branch}
        )

        data = {
            "message": params.get("commit_message", "Update file via Agentical"),
            "content": base64.b64encode(params.get("content", "").encode()).decode(),
            "sha": current_file.get("sha", "abc123def456"),  # Mock SHA
            "branch": request.branch
        }

        response = await self._make_github_request(
            "PUT",
            f"repos/{request.repository}/contents/{request.path}",
            data=data
        )

        return {
            "success": True,
            "file_info": response,
            "path": request.path,
            "operation": "update_file"
        }

    async def _get_branches(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Get repository branches."""
        response = await self._make_github_request(
            "GET",
            f"repos/{request.repository}/branches"
        )

        return {
            "success": True,
            "branches": response,
            "operation": "get_branches"
        }

    async def _create_branch(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Create a new branch."""
        params = request.parameters or {}

        # Get the SHA of the source branch
        source_branch = params.get("source_branch", "main")
        branch_info = await self._make_github_request(
            "GET",
            f"repos/{request.repository}/git/ref/heads/{source_branch}"
        )

        data = {
            "ref": f"refs/heads/{params.get('branch_name', 'new-branch')}",
            "sha": branch_info.get("object", {}).get("sha", "abc123def456")
        }

        response = await self._make_github_request(
            "POST",
            f"repos/{request.repository}/git/refs",
            data=data
        )

        return {
            "success": True,
            "branch_info": response,
            "operation": "create_branch"
        }

    async def _protect_branch(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Protect a branch with rules."""
        params = request.parameters or {}

        protection_data = {
            "required_status_checks": params.get("required_status_checks"),
            "enforce_admins": params.get("enforce_admins", False),
            "required_pull_request_reviews": params.get("required_pull_request_reviews"),
            "restrictions": params.get("restrictions")
        }

        response = await self._make_github_request(
            "PUT",
            f"repos/{request.repository}/branches/{request.branch}/protection",
            data=protection_data
        )

        return {
            "success": True,
            "protection_info": response,
            "branch": request.branch,
            "operation": "protect_branch"
        }

    async def _execute_generic_action(self, request: RepositoryRequest) -> Dict[str, Any]:
        """Execute a generic repository action."""
        return {
            "success": True,
            "action": request.action,
            "message": f"Generic action {request.action} executed successfully",
            "operation": "generic_action"
        }

    async def manage_pull_request(self, request: PullRequestRequest) -> Dict[str, Any]:
        """
        Manage pull request operations.

        Args:
            request: Pull request request

        Returns:
            Pull request operation results
        """
        with logfire.span(
            "GitHubAgent.manage_pull_request",
            agent_id=self.agent_id,
            repository=request.repository,
            action=request.action
        ):
            try:
                if request.action not in self.supported_actions.get("pull_request", []):
                    raise ValidationError(f"Unsupported pull request action: {request.action}")

                if request.action == "create":
                    result = await self._create_pull_request(request)
                elif request.action == "review":
                    result = await self._review_pull_request(request)
                elif request.action == "merge":
                    result = await self._merge_pull_request(request)
                elif request.action == "close":
                    result = await self._close_pull_request(request)
                else:
                    result = {"success": True, "action": request.action}

                logfire.info(
                    "Pull request operation completed",
                    agent_id=self.agent_id,
                    repository=request.repository,
                    action=request.action
                )

                return result

            except Exception as e:
                logfire.error(
                    "Pull request operation failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Pull request operation failed: {str(e)}")

    async def _create_pull_request(self, request: PullRequestRequest) -> Dict[str, Any]:
        """Create a new pull request."""
        data = {
            "title": request.title,
            "body": request.body,
            "head": request.head_branch,
            "base": request.base_branch
        }

        response = await self._make_github_request(
            "POST",
            f"repos/{request.repository}/pulls",
            data=data
        )

        return {
            "success": True,
            "pull_request": {
                "number": 123,  # Mock PR number
                "title": request.title,
                "state": "open",
                "html_url": f"https://github.com/{request.repository}/pull/123"
            },
            "operation": "create_pull_request"
        }

    async def _review_pull_request(self, request: PullRequestRequest) -> Dict[str, Any]:
        """Review a pull request."""
        data = {
            "body": request.review_body,
            "event": request.review_action.upper()  # APPROVE, REQUEST_CHANGES, COMMENT
        }

        response = await self._make_github_request(
            "POST",
            f"repos/{request.repository}/pulls/{request.pr_number}/reviews",
            data=data
        )

        return {
            "success": True,
            "review": {
                "id": 456,  # Mock review ID
                "state": request.review_action,
                "body": request.review_body
            },
            "operation": "review_pull_request"
        }

    async def _merge_pull_request(self, request: PullRequestRequest) -> Dict[str, Any]:
        """Merge a pull request."""
        data = {
            "commit_title": f"Merge pull request #{request.pr_number}",
            "merge_method": "merge"  # merge, squash, rebase
        }

        response = await self._make_github_request(
            "PUT",
            f"repos/{request.repository}/pulls/{request.pr_number}/merge",
            data=data
        )

        return {
            "success": True,
            "merge": {
                "sha": "abc123def456789",
                "merged": True,
                "message": "Pull request merged successfully"
            },
            "operation": "merge_pull_request"
        }

    async def _close_pull_request(self, request: PullRequestRequest) -> Dict[str, Any]:
        """Close a pull request."""
        data = {"state": "closed"}

        response = await self._make_github_request(
            "PATCH",
            f"repos/{request.repository}/pulls/{request.pr_number}",
            data=data
        )

        return {
            "success": True,
            "pull_request": {
                "number": request.pr_number,
                "state": "closed"
            },
            "operation": "close_pull_request"
        }

    async def manage_issues(self, request: IssueRequest) -> Dict[str, Any]:
        """
        Manage GitHub issues.

        Args:
            request: Issue request

        Returns:
            Issue operation results
        """
        with logfire.span(
            "GitHubAgent.manage_issues",
            agent_id=self.agent_id,
            repository=request.repository,
            action=request.action
        ):
            try:
                if request.action not in self.supported_actions.get("issue", []):
                    raise ValidationError(f"Unsupported issue action: {request.action}")

                # Mock issue operations
                result = {
                    "success": True,
                    "issue": {
                        "number": request.issue_number or 789,
                        "title": request.title,
                        "body": request.body,
                        "state": "open",
                        "labels": request.labels or [],
                        "assignees": request.assignees or [],
                        "html_url": f"https://github.com/{request.repository}/issues/{request.issue_number or 789}"
                    },
                    "operation": f"{request.action}_issue"
                }

                logfire.info(
                    "Issue operation completed",
                    agent_id=self.agent_id,
                    repository=request.repository,
                    action=request.action
                )

                return result

            except Exception as e:
                logfire.error(
                    "Issue operation failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Issue operation failed: {str(e)}")

    async def manage_workflows(self, request: WorkflowRequest) -> Dict[str, Any]:
        """
        Manage GitHub Actions workflows.

        Args:
            request: Workflow request

        Returns:
            Workflow operation results
        """
        with logfire.span(
            "GitHubAgent.manage_workflows",
            agent_id=self.agent_id,
            repository=request.repository,
            action=request.action
        ):
            try:
                if request.action not in self.supported_actions.get("workflow", []):
                    raise ValidationError(f"Unsupported workflow action: {request.action}")

                # Mock workflow operations
                result = {
                    "success": True,
                    "workflow": {
                        "id": 123,
                        "name": request.workflow_name,
                        "state": "active",
                        "run_id": request.run_id or 456
                    },
                    "operation": f"{request.action}_workflow"
                }

                logfire.info(
                    "Workflow operation completed",
                    agent_id=self.agent_id,
                    repository=request.repository,
                    action=request.action
                )

                return result

            except Exception as e:
                logfire.error(
                    "Workflow operation failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Workflow operation failed: {str(e)}")

    async def analyze_code(self, request: CodeAnalysisRequest) -> Dict[str, Any]:
        """
        Perform code analysis on repository.

        Args:
            request: Code analysis request

        Returns:
            Code analysis results
        """
        with logfire.span(
            "GitHubAgent.analyze_code",
            agent_id=self.agent_id,
            repository=request.repository,
            analysis_type=request.analysis_type
        ):
            try:
                # Mock code analysis results
                analysis_results = {
                    "analysis_type": request.analysis_type,
                    "repository": request.repository,
                    "ref": request.ref,
                    "findings": [],
                    "summary": {},
                    "timestamp": datetime.utcnow().isoformat()
                }

                if request.analysis_type == "security":
                    analysis_results["findings"] = [
                        {
                            "rule_id": "SECURITY-001",
                            "severity": "high",
                            "message": "Potential SQL injection vulnerability",
                            "file": "src/database.py",
                            "line": 42
                        }
                    ]
                    analysis_results["summary"] = {
                        "total_findings": 1,
                        "high_severity": 1,
                        "medium_severity": 0,
                        "low_severity": 0
                    }

                elif request.analysis_type == "quality":
                    analysis_results["findings"] = [
                        {
                            "rule_id": "QUALITY-001",
                            "severity": "medium",
                            "message": "Function complexity too high",
                            "file": "src/utils.py",
                            "line": 15
                        }
                    ]
                    analysis_results["summary"] = {
                        "quality_score": 8.5,
                        "maintainability": "A",
                        "technical_debt": "2h 30m"
                    }

                elif request.analysis_type == "dependencies":
                    analysis_results["findings"] = [
                        {
                            "package": "requests",
                            "current_version": "2.25.1",
                            "latest_version": "2.31.0",
                            "vulnerabilities": 2
                        }
                    ]
                    analysis_results["summary"] = {
                        "total_dependencies": 45,
                        "outdated": 12,
                        "vulnerable": 3
                    }

                logfire.info(
                    "Code analysis completed",
                    agent_id=self.agent_id,
                    repository=request.repository,
                    analysis_type=request.analysis_type,
                    findings_count=len(analysis_results["findings"])
                )

                return {
                    "success": True,
                    "analysis_results": analysis_results,
                    "operation": "analyze_code"
                }

            except Exception as e:
                logfire.error(
                    "Code analysis failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Code analysis failed: {str(e)}")

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for GitHub agent."""
        return {
            "api_base_url": "https://api.github.com",
            "default_branch": "main",
            "auto_merge_enabled": False,
            "review_required": True,
            "status_checks_required": True,
            "branch_protection_enabled": True,
            "security_scanning_enabled": True,
            "dependency_updates_enabled": True,
            "issue_auto_assignment": False,
            "pr_auto_review": False,
            "workflow_timeout": 3600,  # 1 hour
            "max_file_size": 1048576,  # 1MB
            "supported_actions": self.supported_actions,
            "code_analysis_tools": self.code_analysis_tools,
            "webhook_events": self.webhook_events
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_fields = ["api_base_url", "default_branch"]

        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required configuration field: {field}")

        # Validate API URL
        if not config.get("api_base_url", "").startswith("https://"):
            raise ValidationError("api_base_url must be HTTPS")

        # Validate timeout
        if config.get("workflow_timeout", 0) <= 0:
            raise ValidationError("workflow_timeout must be positive")

        return True
