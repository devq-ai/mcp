"""
Task 4.4 Custom Agent Classes Implementation Validation

This script validates the implementation of Task 4.4 custom agent classes:
- CodifierAgent (documentation/logging)
- IOAgent (inspector_observer)
- PlaybookAgent (strategic execution)
- SuperAgent (meta-coordination)

Validation includes:
- Agent class structure and inheritance
- Required methods and capabilities
- Database model integration
- Configuration validation
- Basic functionality testing
"""

import asyncio
import sys
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logfire
from agentical.agents.codifier_agent import (
    CodifierAgent, DocumentationRequest, LogStructuringRequest,
    KnowledgeCodeRequest, DocumentationType, LogFormat
)
from agentical.agents.io_agent import (
    IOAgent, InspectionRequest, MonitoringRequest, ObservationRequest,
    MonitoringScope, ObservationType, AlertSeverity
)
from agentical.agents.playbook_agent import (
    PlaybookAgent, PlaybookExecutionRequest, PlaybookCreationRequest,
    PlaybookAnalysisRequest, ExecutionMode, ValidationLevel
)
from agentical.agents.super_agent import SuperAgent
from agentical.db.models.agent import AgentType
from agentical.db.models.playbook import PlaybookCategory


class Task44Validator:
    """Validator for Task 4.4 custom agent implementation."""

    def __init__(self):
        self.validation_results = {
            "task_4_4_validation": {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "unknown",
                "agents_validated": 0,
                "total_agents": 4,
                "validation_details": {}
            }
        }

    async def run_validation(self) -> Dict[str, Any]:
        """Run complete Task 4.4 validation."""

        print("🔍 Starting Task 4.4 Custom Agent Classes Validation...")
        print("=" * 60)

        try:
            # Validate each custom agent
            agents_to_validate = [
                ("CodifierAgent", self.validate_codifier_agent),
                ("IOAgent", self.validate_io_agent),
                ("PlaybookAgent", self.validate_playbook_agent),
                ("SuperAgent", self.validate_super_agent)
            ]

            for agent_name, validator_func in agents_to_validate:
                print(f"\n📋 Validating {agent_name}...")
                result = await validator_func()
                self.validation_results["task_4_4_validation"]["validation_details"][agent_name] = result

                if result["status"] == "passed":
                    self.validation_results["task_4_4_validation"]["agents_validated"] += 1
                    print(f"✅ {agent_name} validation PASSED")
                else:
                    print(f"❌ {agent_name} validation FAILED")
                    for error in result.get("errors", []):
                        print(f"   - {error}")

            # Calculate overall status
            validated_count = self.validation_results["task_4_4_validation"]["agents_validated"]
            total_count = self.validation_results["task_4_4_validation"]["total_agents"]

            if validated_count == total_count:
                self.validation_results["task_4_4_validation"]["overall_status"] = "passed"
                print(f"\n🎉 Task 4.4 Implementation: ALL {total_count} AGENTS VALIDATED SUCCESSFULLY!")
            else:
                self.validation_results["task_4_4_validation"]["overall_status"] = "failed"
                print(f"\n⚠️  Task 4.4 Implementation: {validated_count}/{total_count} agents validated")

            # Print summary
            self.print_validation_summary()

        except Exception as e:
            self.validation_results["task_4_4_validation"]["overall_status"] = "error"
            self.validation_results["task_4_4_validation"]["error"] = str(e)
            print(f"\n💥 Validation failed with error: {str(e)}")

        return self.validation_results

    async def validate_codifier_agent(self) -> Dict[str, Any]:
        """Validate CodifierAgent implementation."""

        result = {
            "status": "unknown",
            "checks": {},
            "errors": [],
            "capabilities_count": 0,
            "request_types": []
        }

        try:
            # Test agent instantiation
            agent = CodifierAgent(agent_id="test_codifier")
            result["checks"]["instantiation"] = True

            # Check agent type
            if agent.agent_type == AgentType.CODIFIER_AGENT:
                result["checks"]["agent_type"] = True
            else:
                result["errors"].append(f"Wrong agent type: {agent.agent_type}")

            # Check capabilities
            capabilities = agent.get_capabilities()
            result["capabilities_count"] = len(capabilities)
            expected_capabilities = [
                "documentation_generation", "code_documentation", "log_analysis",
                "knowledge_codification", "content_standardization"
            ]

            capabilities_check = all(cap in capabilities for cap in expected_capabilities)
            result["checks"]["capabilities"] = capabilities_check
            if not capabilities_check:
                missing = [cap for cap in expected_capabilities if cap not in capabilities]
                result["errors"].append(f"Missing capabilities: {missing}")

            # Test request models
            try:
                doc_request = DocumentationRequest(
                    documentation_type=DocumentationType.API_DOCUMENTATION,
                    content="Test API documentation"
                )
                result["request_types"].append("DocumentationRequest")

                log_request = LogStructuringRequest(
                    log_data="test log data",
                    source_format=LogFormat.JSON,
                    target_format=LogFormat.STRUCTURED
                )
                result["request_types"].append("LogStructuringRequest")

                knowledge_request = KnowledgeCodeRequest(
                    knowledge_source="test knowledge",
                    knowledge_type="technical"
                )
                result["request_types"].append("KnowledgeCodeRequest")

                result["checks"]["request_models"] = True
            except Exception as e:
                result["errors"].append(f"Request model validation failed: {str(e)}")

            # Test configuration
            try:
                config = agent.get_default_configuration()
                await agent.validate_configuration(config)
                result["checks"]["configuration"] = True
            except Exception as e:
                result["errors"].append(f"Configuration validation failed: {str(e)}")

            # Test core execution (simple dry run)
            try:
                test_request = DocumentationRequest(
                    documentation_type=DocumentationType.README,
                    content="Test documentation generation"
                )

                # Mock execution context
                context = {"test": True, "dry_run": True}
                execution_result = await agent._execute_core_logic(test_request, context)

                if "documentation_type" in execution_result:
                    result["checks"]["core_execution"] = True
                else:
                    result["errors"].append("Core execution did not return expected result")

            except Exception as e:
                result["errors"].append(f"Core execution test failed: {str(e)}")

            # Determine overall status
            if len(result["errors"]) == 0 and all(result["checks"].values()):
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Agent validation error: {str(e)}")

        return result

    async def validate_io_agent(self) -> Dict[str, Any]:
        """Validate IOAgent implementation."""

        result = {
            "status": "unknown",
            "checks": {},
            "errors": [],
            "capabilities_count": 0,
            "request_types": []
        }

        try:
            # Test agent instantiation
            agent = IOAgent(agent_id="test_io")
            result["checks"]["instantiation"] = True

            # Check agent type
            if agent.agent_type == AgentType.IO_AGENT:
                result["checks"]["agent_type"] = True
            else:
                result["errors"].append(f"Wrong agent type: {agent.agent_type}")

            # Check capabilities
            capabilities = agent.get_capabilities()
            result["capabilities_count"] = len(capabilities)
            expected_capabilities = [
                "system_monitoring", "application_monitoring", "health_checks",
                "anomaly_detection", "real_time_observation"
            ]

            capabilities_check = all(cap in capabilities for cap in expected_capabilities)
            result["checks"]["capabilities"] = capabilities_check
            if not capabilities_check:
                missing = [cap for cap in expected_capabilities if cap not in capabilities]
                result["errors"].append(f"Missing capabilities: {missing}")

            # Test request models
            try:
                inspection_request = InspectionRequest(
                    scope=MonitoringScope.SYSTEM,
                    targets=["localhost"],
                    observation_type=ObservationType.HEALTH_CHECK
                )
                result["request_types"].append("InspectionRequest")

                monitoring_request = MonitoringRequest(
                    scope=MonitoringScope.APPLICATION,
                    duration_minutes=1
                )
                result["request_types"].append("MonitoringRequest")

                observation_request = ObservationRequest(
                    observation_type=ObservationType.PERFORMANCE_METRICS,
                    target="test_service"
                )
                result["request_types"].append("ObservationRequest")

                result["checks"]["request_models"] = True
            except Exception as e:
                result["errors"].append(f"Request model validation failed: {str(e)}")

            # Test configuration
            try:
                config = agent.get_default_configuration()
                await agent.validate_configuration(config)
                result["checks"]["configuration"] = True
            except Exception as e:
                result["errors"].append(f"Configuration validation failed: {str(e)}")

            # Test core execution (simple dry run)
            try:
                test_request = InspectionRequest(
                    scope=MonitoringScope.SYSTEM,
                    targets=["test_target"],
                    observation_type=ObservationType.HEALTH_CHECK
                )

                context = {"test": True, "dry_run": True}
                execution_result = await agent._execute_core_logic(test_request, context)

                if "inspection_id" in execution_result:
                    result["checks"]["core_execution"] = True
                else:
                    result["errors"].append("Core execution did not return expected result")

            except Exception as e:
                result["errors"].append(f"Core execution test failed: {str(e)}")

            # Determine overall status
            if len(result["errors"]) == 0 and all(result["checks"].values()):
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Agent validation error: {str(e)}")

        return result

    async def validate_playbook_agent(self) -> Dict[str, Any]:
        """Validate PlaybookAgent implementation."""

        result = {
            "status": "unknown",
            "checks": {},
            "errors": [],
            "capabilities_count": 0,
            "request_types": [],
            "template_count": 0
        }

        try:
            # Test agent instantiation
            agent = PlaybookAgent(agent_id="test_playbook")
            result["checks"]["instantiation"] = True

            # Check agent type
            if agent.agent_type == AgentType.PLAYBOOK_AGENT:
                result["checks"]["agent_type"] = True
            else:
                result["errors"].append(f"Wrong agent type: {agent.agent_type}")

            # Check capabilities
            capabilities = agent.get_capabilities()
            result["capabilities_count"] = len(capabilities)
            expected_capabilities = [
                "playbook_execution", "playbook_creation", "strategic_planning",
                "workflow_orchestration", "multi_agent_coordination"
            ]

            capabilities_check = all(cap in capabilities for cap in expected_capabilities)
            result["checks"]["capabilities"] = capabilities_check
            if not capabilities_check:
                missing = [cap for cap in expected_capabilities if cap not in capabilities]
                result["errors"].append(f"Missing capabilities: {missing}")

            # Check playbook templates
            result["template_count"] = len(agent.playbook_templates)
            expected_templates = ["incident_response", "deployment", "troubleshooting"]
            template_check = all(template in agent.playbook_templates for template in expected_templates)
            result["checks"]["templates"] = template_check
            if not template_check:
                missing = [t for t in expected_templates if t not in agent.playbook_templates]
                result["errors"].append(f"Missing templates: {missing}")

            # Test request models
            try:
                execution_request = PlaybookExecutionRequest(
                    playbook_id="test_playbook",
                    execution_mode=ExecutionMode.SEQUENTIAL
                )
                result["request_types"].append("PlaybookExecutionRequest")

                creation_request = PlaybookCreationRequest(
                    name="Test Playbook",
                    description="Test playbook description",
                    category=PlaybookCategory.TESTING,
                    steps=[{"name": "Test Step", "type": "action"}]
                )
                result["request_types"].append("PlaybookCreationRequest")

                analysis_request = PlaybookAnalysisRequest(
                    analysis_type="performance"
                )
                result["request_types"].append("PlaybookAnalysisRequest")

                result["checks"]["request_models"] = True
            except Exception as e:
                result["errors"].append(f"Request model validation failed: {str(e)}")

            # Test configuration
            try:
                config = agent.get_default_configuration()
                await agent.validate_configuration(config)
                result["checks"]["configuration"] = True
            except Exception as e:
                result["errors"].append(f"Configuration validation failed: {str(e)}")

            # Test core execution (simple dry run)
            try:
                test_request = PlaybookCreationRequest(
                    name="Test Validation Playbook",
                    description="Test playbook for validation",
                    category=PlaybookCategory.TESTING,
                    steps=[
                        {"name": "Step 1", "type": "action", "action": "test"},
                        {"name": "Step 2", "type": "validation", "validation_type": "basic"}
                    ]
                )

                context = {"test": True, "dry_run": True}
                execution_result = await agent._execute_core_logic(test_request, context)

                if "playbook_id" in execution_result:
                    result["checks"]["core_execution"] = True
                else:
                    result["errors"].append("Core execution did not return expected result")

            except Exception as e:
                result["errors"].append(f"Core execution test failed: {str(e)}")

            # Determine overall status
            if len(result["errors"]) == 0 and all(result["checks"].values()):
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Agent validation error: {str(e)}")

        return result

    async def validate_super_agent(self) -> Dict[str, Any]:
        """Validate SuperAgent implementation."""

        result = {
            "status": "unknown",
            "checks": {},
            "errors": [],
            "capabilities_count": 0
        }

        try:
            # Test agent instantiation
            agent = SuperAgent()
            result["checks"]["instantiation"] = True

            # Check if SuperAgent exists and has basic structure
            if hasattr(agent, 'agent_type'):
                result["checks"]["agent_type"] = True
            else:
                result["errors"].append("SuperAgent missing agent_type attribute")

            # Check if SuperAgent has coordination methods
            coordination_methods = [
                '_coordinate_agents', '_intelligent_routing',
                '_knowledge_synthesis', '_multimodal_coordination'
            ]

            missing_methods = []
            for method in coordination_methods:
                if not hasattr(agent, method):
                    missing_methods.append(method)

            if not missing_methods:
                result["checks"]["coordination_methods"] = True
            else:
                result["errors"].append(f"Missing coordination methods: {missing_methods}")

            # Check if SuperAgent can handle requests
            if hasattr(agent, '_execute_operation'):
                result["checks"]["execution_method"] = True
            else:
                result["errors"].append("SuperAgent missing _execute_operation method")

            # Determine overall status
            if len(result["errors"]) == 0 and all(result["checks"].values()):
                result["status"] = "passed"
            else:
                result["status"] = "failed"

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Agent validation error: {str(e)}")

        return result

    def print_validation_summary(self):
        """Print detailed validation summary."""

        print("\n" + "=" * 60)
        print("📊 TASK 4.4 VALIDATION SUMMARY")
        print("=" * 60)

        overall_status = self.validation_results["task_4_4_validation"]["overall_status"]
        agents_validated = self.validation_results["task_4_4_validation"]["agents_validated"]
        total_agents = self.validation_results["task_4_4_validation"]["total_agents"]

        print(f"Overall Status: {'✅ PASSED' if overall_status == 'passed' else '❌ FAILED'}")
        print(f"Agents Validated: {agents_validated}/{total_agents}")

        print("\nAgent Details:")
        for agent_name, details in self.validation_results["task_4_4_validation"]["validation_details"].items():
            status_icon = "✅" if details["status"] == "passed" else "❌"
            print(f"  {status_icon} {agent_name}: {details['status'].upper()}")

            if "capabilities_count" in details:
                print(f"     - Capabilities: {details['capabilities_count']}")
            if "request_types" in details:
                print(f"     - Request Types: {len(details['request_types'])}")
            if "template_count" in details:
                print(f"     - Templates: {details['template_count']}")
            if details.get("errors"):
                print(f"     - Errors: {len(details['errors'])}")

        print("\nTask 4.4 Custom Agent Classes Implementation:")
        print("✓ CodifierAgent - Documentation & logging specialist")
        print("✓ IOAgent - Inspector observer specialist")
        print("✓ PlaybookAgent - Strategic execution specialist")
        print("✓ SuperAgent - Meta-coordination specialist")

        if overall_status == "passed":
            print("\n🎉 Task 4.4 Implementation COMPLETE!")
            print("All 4 custom agent classes successfully implemented and validated.")
        else:
            print(f"\n⚠️  Task 4.4 Implementation INCOMPLETE!")
            print(f"Issues found with {total_agents - agents_validated} agent(s).")

    def save_validation_report(self, filename: str = None):
        """Save validation results to file."""

        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"task_4_4_validation_report_{timestamp}.json"

        report_path = project_root / filename

        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        print(f"\n💾 Validation report saved to: {report_path}")


async def main():
    """Main validation function."""

    # Configure logging
    logfire.configure()

    validator = Task44Validator()

    try:
        results = await validator.run_validation()
        validator.save_validation_report()

        # Exit with appropriate code
        if results["task_4_4_validation"]["overall_status"] == "passed":
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Validation script failed: {str(e)}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
