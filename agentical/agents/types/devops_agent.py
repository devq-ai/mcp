"""
DevOps Agent Implementation for Agentical Framework

This module provides the DevOpsAgent implementation for infrastructure management,
deployment automation, operations monitoring, and CI/CD pipeline orchestration.

Features:
- Infrastructure provisioning and management
- CI/CD pipeline automation
- Container orchestration (Docker, Kubernetes)
- Cloud resource management
- Monitoring and alerting setup
- Security scanning and compliance
- Performance optimization
- Incident response automation
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
import yaml

import logfire
from pydantic import BaseModel, Field

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.core.exceptions import AgentExecutionError, ValidationError
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase


class InfrastructureRequest(BaseModel):
    """Request model for infrastructure management tasks."""
    provider: str = Field(..., description="Cloud provider (aws, gcp, azure, local)")
    action: str = Field(..., description="Action to perform (create, update, destroy, status)")
    resource_type: str = Field(..., description="Type of resource to manage")
    configuration: Dict[str, Any] = Field(..., description="Resource configuration")
    environment: str = Field(default="development", description="Target environment")
    region: Optional[str] = Field(default=None, description="Target region")
    tags: Optional[Dict[str, str]] = Field(default=None, description="Resource tags")


class DeploymentRequest(BaseModel):
    """Request model for application deployment tasks."""
    application: str = Field(..., description="Application name")
    version: str = Field(..., description="Version to deploy")
    environment: str = Field(..., description="Target environment")
    deployment_strategy: str = Field(default="rolling", description="Deployment strategy")
    configuration: Dict[str, Any] = Field(..., description="Deployment configuration")
    rollback_enabled: bool = Field(default=True, description="Enable automatic rollback")
    health_checks: Optional[List[Dict[str, Any]]] = Field(default=None, description="Health check definitions")


class PipelineRequest(BaseModel):
    """Request model for CI/CD pipeline operations."""
    pipeline_name: str = Field(..., description="Pipeline name")
    action: str = Field(..., description="Action (create, run, stop, status)")
    repository: str = Field(..., description="Source repository")
    branch: str = Field(default="main", description="Source branch")
    stages: List[Dict[str, Any]] = Field(..., description="Pipeline stages")
    environment_variables: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    secrets: Optional[Dict[str, str]] = Field(default=None, description="Secret references")


class MonitoringRequest(BaseModel):
    """Request model for monitoring and alerting setup."""
    target: str = Field(..., description="Monitoring target")
    metrics: List[str] = Field(..., description="Metrics to monitor")
    alert_rules: List[Dict[str, Any]] = Field(..., description="Alert rule definitions")
    notification_channels: List[str] = Field(..., description="Notification channels")
    dashboard_config: Optional[Dict[str, Any]] = Field(default=None, description="Dashboard configuration")


class SecurityScanRequest(BaseModel):
    """Request model for security scanning tasks."""
    scan_type: str = Field(..., description="Type of scan (vulnerability, compliance, configuration)")
    target: str = Field(..., description="Scan target")
    policies: Optional[List[str]] = Field(default=None, description="Security policies to enforce")
    severity_threshold: str = Field(default="medium", description="Minimum severity to report")
    output_format: str = Field(default="json", description="Output format")


class DevOpsAgent(EnhancedBaseAgent[InfrastructureRequest, Dict[str, Any]]):
    """
    Specialized agent for DevOps, infrastructure, and operations management.

    Capabilities:
    - Infrastructure provisioning and management
    - CI/CD pipeline automation
    - Container orchestration
    - Cloud resource management
    - Monitoring and alerting
    - Security scanning
    - Performance optimization
    - Incident response automation
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "DevOpsAgent",
        description: str = "Specialized agent for DevOps and infrastructure management",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.DEVOPS_AGENT,
            **kwargs
        )

        # DevOps specific configuration
        self.supported_providers = {
            "cloud": ["aws", "gcp", "azure", "digitalocean", "linode"],
            "container": ["docker", "kubernetes", "docker-compose"],
            "ci_cd": ["github-actions", "gitlab-ci", "jenkins", "circleci", "azure-devops"],
            "monitoring": ["prometheus", "grafana", "datadog", "newrelic", "cloudwatch"],
            "security": ["snyk", "aqua", "twistlock", "qualys", "nessus"]
        }

        self.infrastructure_tools = {
            "terraform": {
                "commands": ["init", "plan", "apply", "destroy", "validate"],
                "providers": ["aws", "gcp", "azure", "kubernetes"]
            },
            "ansible": {
                "commands": ["playbook", "inventory", "vault", "galaxy"],
                "modules": ["cloud", "system", "network", "security"]
            },
            "kubernetes": {
                "commands": ["apply", "delete", "get", "describe", "logs"],
                "resources": ["pod", "service", "deployment", "configmap", "secret"]
            },
            "docker": {
                "commands": ["build", "run", "push", "pull", "compose"],
                "registries": ["dockerhub", "ecr", "gcr", "acr"]
            }
        }

        self.deployment_strategies = {
            "rolling": "Gradual replacement of instances",
            "blue_green": "Switch between two identical environments",
            "canary": "Gradual traffic shift to new version",
            "recreate": "Stop all instances and create new ones",
            "a_b_testing": "Split traffic between versions for testing"
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "infrastructure_provisioning",
            "infrastructure_management",
            "application_deployment",
            "ci_cd_automation",
            "container_orchestration",
            "cloud_resource_management",
            "monitoring_setup",
            "alerting_configuration",
            "security_scanning",
            "compliance_checking",
            "performance_optimization",
            "incident_response",
            "backup_management",
            "disaster_recovery",
            "configuration_management",
            "secret_management",
            "network_configuration",
            "load_balancing",
            "auto_scaling",
            "cost_optimization"
        ]

    async def _execute_core_logic(
        self,
        request: InfrastructureRequest,
        correlation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute core DevOps logic.

        Args:
            request: Infrastructure request
            correlation_context: Optional correlation context

        Returns:
            Operation results with status and resource information
        """
        with logfire.span(
            "DevOpsAgent.execute_core_logic",
            agent_id=self.agent_id,
            provider=request.provider,
            action=request.action,
            resource_type=request.resource_type
        ):
            self.logger.log_operation(
                OperationType.AGENT_EXECUTION,
                AgentPhase.EXECUTION,
                {
                    "provider": request.provider,
                    "action": request.action,
                    "resource_type": request.resource_type,
                    "environment": request.environment
                },
                correlation_context
            )

            try:
                # Validate provider support
                all_providers = [p for providers in self.supported_providers.values() for p in providers]
                if request.provider not in all_providers:
                    raise ValidationError(f"Unsupported provider: {request.provider}")

                # Execute action based on type
                if request.action == "create":
                    result = await self._create_infrastructure(request)
                elif request.action == "update":
                    result = await self._update_infrastructure(request)
                elif request.action == "destroy":
                    result = await self._destroy_infrastructure(request)
                elif request.action == "status":
                    result = await self._get_infrastructure_status(request)
                else:
                    raise ValidationError(f"Unsupported action: {request.action}")

                # Add metadata
                result.update({
                    "provider": request.provider,
                    "action": request.action,
                    "resource_type": request.resource_type,
                    "environment": request.environment,
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_id": self.agent_id
                })

                logfire.info(
                    "DevOps operation completed",
                    agent_id=self.agent_id,
                    action=request.action,
                    success=result.get("success", False)
                )

                return result

            except Exception as e:
                logfire.error(
                    "DevOps operation failed",
                    agent_id=self.agent_id,
                    error=str(e),
                    action=request.action
                )
                raise AgentExecutionError(f"DevOps operation failed: {str(e)}")

    async def _create_infrastructure(self, request: InfrastructureRequest) -> Dict[str, Any]:
        """Create infrastructure resources."""

        # Generate Terraform configuration
        terraform_config = self._generate_terraform_config(request)

        # Create temporary directory for Terraform files
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "main.tf"
            config_file.write_text(terraform_config)

            # Initialize Terraform
            init_result = await self._run_terraform_command(temp_dir, ["init"])
            if not init_result["success"]:
                raise AgentExecutionError(f"Terraform init failed: {init_result['error']}")

            # Plan infrastructure
            plan_result = await self._run_terraform_command(temp_dir, ["plan", "-out=tfplan"])
            if not plan_result["success"]:
                raise AgentExecutionError(f"Terraform plan failed: {plan_result['error']}")

            # Apply infrastructure
            apply_result = await self._run_terraform_command(temp_dir, ["apply", "tfplan"])

            return {
                "success": apply_result["success"],
                "resource_id": f"{request.resource_type}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "configuration": request.configuration,
                "terraform_output": apply_result.get("output", ""),
                "plan_summary": self._parse_terraform_plan(plan_result.get("output", ""))
            }

    async def _update_infrastructure(self, request: InfrastructureRequest) -> Dict[str, Any]:
        """Update existing infrastructure resources."""

        # Generate updated configuration
        terraform_config = self._generate_terraform_config(request)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "main.tf"
            config_file.write_text(terraform_config)

            # Import existing state (simplified)
            # In production, this would properly manage state

            # Plan update
            plan_result = await self._run_terraform_command(temp_dir, ["plan"])

            # Apply update
            apply_result = await self._run_terraform_command(temp_dir, ["apply", "-auto-approve"])

            return {
                "success": apply_result["success"],
                "changes_applied": True,
                "update_summary": self._parse_terraform_plan(plan_result.get("output", "")),
                "terraform_output": apply_result.get("output", "")
            }

    async def _destroy_infrastructure(self, request: InfrastructureRequest) -> Dict[str, Any]:
        """Destroy infrastructure resources."""

        # Generate configuration for destruction
        terraform_config = self._generate_terraform_config(request)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "main.tf"
            config_file.write_text(terraform_config)

            # Plan destruction
            plan_result = await self._run_terraform_command(temp_dir, ["plan", "-destroy"])

            # Destroy infrastructure
            destroy_result = await self._run_terraform_command(temp_dir, ["destroy", "-auto-approve"])

            return {
                "success": destroy_result["success"],
                "resources_destroyed": True,
                "destruction_summary": self._parse_terraform_plan(plan_result.get("output", "")),
                "terraform_output": destroy_result.get("output", "")
            }

    async def _get_infrastructure_status(self, request: InfrastructureRequest) -> Dict[str, Any]:
        """Get status of infrastructure resources."""

        # This would query actual infrastructure state
        # For now, return mock status
        return {
            "success": True,
            "status": "running",
            "resources": [
                {
                    "type": request.resource_type,
                    "name": f"{request.resource_type}-instance",
                    "status": "healthy",
                    "region": request.region or "us-east-1",
                    "tags": request.tags or {}
                }
            ],
            "health_checks": {
                "overall": "healthy",
                "components": {
                    "compute": "healthy",
                    "network": "healthy",
                    "storage": "healthy"
                }
            }
        }

    def _generate_terraform_config(self, request: InfrastructureRequest) -> str:
        """Generate Terraform configuration from request."""

        config_template = f"""
terraform {{
  required_providers {{
    {request.provider} = {{
      source = "hashicorp/{request.provider}"
      version = "~> 5.0"
    }}
  }}
}}

provider "{request.provider}" {{
  region = "{request.region or 'us-east-1'}"
}}

resource "{request.provider}_{request.resource_type}" "main" {{
  {self._format_terraform_attributes(request.configuration)}

  tags = {{
    Environment = "{request.environment}"
    ManagedBy = "AgenticalDevOps"
    {self._format_terraform_tags(request.tags or {})}
  }}
}}

output "resource_id" {{
  value = {request.provider}_{request.resource_type}.main.id
}}

output "resource_arn" {{
  value = {request.provider}_{request.resource_type}.main.arn
}}
"""
        return config_template.strip()

    def _format_terraform_attributes(self, config: Dict[str, Any]) -> str:
        """Format configuration attributes for Terraform."""
        lines = []
        for key, value in config.items():
            if isinstance(value, str):
                lines.append(f'  {key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f'  {key} = {str(value).lower()}')
            elif isinstance(value, (int, float)):
                lines.append(f'  {key} = {value}')
            elif isinstance(value, list):
                formatted_list = json.dumps(value)
                lines.append(f'  {key} = {formatted_list}')
            elif isinstance(value, dict):
                formatted_dict = json.dumps(value)
                lines.append(f'  {key} = {formatted_dict}')
        return '\n'.join(lines)

    def _format_terraform_tags(self, tags: Dict[str, str]) -> str:
        """Format tags for Terraform configuration."""
        return '\n    '.join([f'{key} = "{value}"' for key, value in tags.items()])

    async def _run_terraform_command(self, work_dir: str, args: List[str]) -> Dict[str, Any]:
        """Run Terraform command in specified directory."""
        try:
            cmd = ["terraform"] + args

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "output": stdout.decode('utf-8') if stdout else "",
                "error": stderr.decode('utf-8') if stderr else "",
                "exit_code": process.returncode
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "exit_code": -1
            }

    def _parse_terraform_plan(self, plan_output: str) -> Dict[str, Any]:
        """Parse Terraform plan output for summary."""
        summary = {
            "resources_to_add": 0,
            "resources_to_change": 0,
            "resources_to_destroy": 0,
            "changes": []
        }

        # Simplified parsing - in production, use proper Terraform JSON output
        lines = plan_output.split('\n')
        for line in lines:
            if "will be created" in line:
                summary["resources_to_add"] += 1
                summary["changes"].append({"action": "create", "resource": line.split()[0]})
            elif "will be updated" in line:
                summary["resources_to_change"] += 1
                summary["changes"].append({"action": "update", "resource": line.split()[0]})
            elif "will be destroyed" in line:
                summary["resources_to_destroy"] += 1
                summary["changes"].append({"action": "destroy", "resource": line.split()[0]})

        return summary

    async def deploy_application(self, request: DeploymentRequest) -> Dict[str, Any]:
        """
        Deploy application using specified strategy.

        Args:
            request: Deployment request

        Returns:
            Deployment results with status and rollback information
        """
        with logfire.span(
            "DevOpsAgent.deploy_application",
            agent_id=self.agent_id,
            application=request.application,
            version=request.version,
            strategy=request.deployment_strategy
        ):
            try:
                # Validate deployment strategy
                if request.deployment_strategy not in self.deployment_strategies:
                    raise ValidationError(f"Unsupported deployment strategy: {request.deployment_strategy}")

                # Pre-deployment checks
                pre_check_result = await self._run_pre_deployment_checks(request)
                if not pre_check_result["success"]:
                    raise AgentExecutionError(f"Pre-deployment checks failed: {pre_check_result['error']}")

                # Execute deployment
                deployment_result = await self._execute_deployment(request)

                # Post-deployment verification
                if deployment_result["success"]:
                    verification_result = await self._verify_deployment(request)
                    deployment_result["verification"] = verification_result

                    if not verification_result["success"] and request.rollback_enabled:
                        rollback_result = await self._rollback_deployment(request)
                        deployment_result["rollback"] = rollback_result

                logfire.info(
                    "Application deployment completed",
                    agent_id=self.agent_id,
                    application=request.application,
                    success=deployment_result["success"]
                )

                return deployment_result

            except Exception as e:
                logfire.error(
                    "Application deployment failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Application deployment failed: {str(e)}")

    async def _run_pre_deployment_checks(self, request: DeploymentRequest) -> Dict[str, Any]:
        """Run pre-deployment validation checks."""
        checks = {
            "image_exists": True,  # Mock check
            "environment_ready": True,
            "resources_available": True,
            "dependencies_satisfied": True
        }

        all_passed = all(checks.values())

        return {
            "success": all_passed,
            "checks": checks,
            "error": None if all_passed else "Some pre-deployment checks failed"
        }

    async def _execute_deployment(self, request: DeploymentRequest) -> Dict[str, Any]:
        """Execute the actual deployment."""
        # Mock deployment execution
        return {
            "success": True,
            "deployment_id": f"deploy-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "strategy": request.deployment_strategy,
            "instances_updated": 3,
            "deployment_time": 120,  # seconds
            "status": "completed"
        }

    async def _verify_deployment(self, request: DeploymentRequest) -> Dict[str, Any]:
        """Verify deployment success through health checks."""
        # Mock verification
        health_checks = request.health_checks or []

        results = []
        for check in health_checks:
            results.append({
                "name": check.get("name", "default"),
                "status": "passed",
                "response_time": 45  # ms
            })

        return {
            "success": True,
            "health_checks": results,
            "overall_health": "healthy"
        }

    async def _rollback_deployment(self, request: DeploymentRequest) -> Dict[str, Any]:
        """Rollback deployment to previous version."""
        # Mock rollback
        return {
            "success": True,
            "rollback_id": f"rollback-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "previous_version": "1.0.0",  # Mock previous version
            "rollback_time": 60  # seconds
        }

    async def setup_monitoring(self, request: MonitoringRequest) -> Dict[str, Any]:
        """
        Set up monitoring and alerting for infrastructure or applications.

        Args:
            request: Monitoring setup request

        Returns:
            Monitoring configuration results
        """
        with logfire.span(
            "DevOpsAgent.setup_monitoring",
            agent_id=self.agent_id,
            target=request.target
        ):
            try:
                # Generate monitoring configuration
                monitoring_config = {
                    "target": request.target,
                    "metrics": request.metrics,
                    "alert_rules": request.alert_rules,
                    "notification_channels": request.notification_channels,
                    "dashboard_url": f"https://monitoring.example.com/dashboard/{request.target}",
                    "setup_status": "completed",
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Mock setup of monitoring infrastructure
                setup_result = {
                    "success": True,
                    "monitoring_config": monitoring_config,
                    "metrics_collected": len(request.metrics),
                    "alerts_configured": len(request.alert_rules),
                    "dashboards_created": 1 if request.dashboard_config else 0
                }

                logfire.info(
                    "Monitoring setup completed",
                    agent_id=self.agent_id,
                    target=request.target,
                    metrics_count=len(request.metrics)
                )

                return setup_result

            except Exception as e:
                logfire.error(
                    "Monitoring setup failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Monitoring setup failed: {str(e)}")

    async def run_security_scan(self, request: SecurityScanRequest) -> Dict[str, Any]:
        """
        Run security scans on infrastructure or applications.

        Args:
            request: Security scan request

        Returns:
            Security scan results with findings and recommendations
        """
        with logfire.span(
            "DevOpsAgent.run_security_scan",
            agent_id=self.agent_id,
            scan_type=request.scan_type,
            target=request.target
        ):
            try:
                # Mock security scan results
                scan_results = {
                    "scan_id": f"scan-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                    "scan_type": request.scan_type,
                    "target": request.target,
                    "severity_threshold": request.severity_threshold,
                    "findings": [
                        {
                            "id": "VULN-001",
                            "severity": "high",
                            "title": "Outdated dependencies detected",
                            "description": "Several dependencies have known security vulnerabilities",
                            "recommendation": "Update to latest versions"
                        },
                        {
                            "id": "CONFIG-001",
                            "severity": "medium",
                            "title": "Insecure configuration detected",
                            "description": "Security group allows unrestricted access",
                            "recommendation": "Restrict access to specific IP ranges"
                        }
                    ],
                    "summary": {
                        "total_findings": 2,
                        "critical": 0,
                        "high": 1,
                        "medium": 1,
                        "low": 0,
                        "info": 0
                    },
                    "compliance_status": "non_compliant",
                    "scan_duration": 180,  # seconds
                    "timestamp": datetime.utcnow().isoformat()
                }

                logfire.info(
                    "Security scan completed",
                    agent_id=self.agent_id,
                    scan_type=request.scan_type,
                    findings_count=scan_results["summary"]["total_findings"]
                )

                return {
                    "success": True,
                    "scan_results": scan_results
                }

            except Exception as e:
                logfire.error(
                    "Security scan failed",
                    agent_id=self.agent_id,
                    error=str(e)
                )
                raise AgentExecutionError(f"Security scan failed: {str(e)}")

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for DevOps agent."""
        return {
            "default_provider": "aws",
            "default_region": "us-east-1",
            "default_environment": "development",
            "terraform_version": "1.5.0",
            "ansible_version": "6.0.0",
            "kubectl_version": "1.28.0",
            "max_deployment_time": 1800,  # 30 minutes
            "auto_rollback_enabled": True,
            "monitoring_enabled": True,
            "security_scanning_enabled": True,
            "backup_enabled": True,
            "cost_optimization_enabled": True,
            "supported_providers": self.supported_providers,
            "infrastructure_tools": self.infrastructure_tools,
            "deployment_strategies": self.deployment_strategies
        }

    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_fields = ["default_provider", "default_region", "max_deployment_time"]

        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required configuration field: {field}")

        # Validate provider
        all_providers = [p for providers in self.supported_providers.values() for p in providers]
        if config.get("default_provider") not in all_providers:
            raise ValidationError(f"Unsupported default provider: {config.get('default_provider')}")

        # Validate timeout
        if config.get("max_deployment_time", 0) <= 0:
            raise ValidationError("max_deployment_time must be positive")

        return True
