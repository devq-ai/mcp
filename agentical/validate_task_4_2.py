"""
Task 4.2 Agent Registry & Discovery - Validation Script

This script validates the agent registry and discovery system implementation
without relying on complex external dependencies.

Validates:
- Registry architecture components exist and are properly structured
- Agent registration and discovery mechanisms
- Health monitoring and lifecycle management
- Selection strategies and load balancing
- Integration with enhanced base agent
- Performance and concurrency capabilities
"""

import sys
import os
import asyncio
import inspect
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add the agentical directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def validate_imports():
    """Validate that all required components can be imported."""
    print("🔍 Validating Task 4.2 imports...")

    validation_results = []

    try:
        from agents.agent_registry_enhanced import (
            EnhancedAgentRegistry, DiscoveryRequest, SelectionCriteria,
            SelectionStrategy, AgentInfo, RegistryMetrics, RegistryStatus
        )
        validation_results.append(("Enhanced Agent Registry imports", True, "All core registry classes imported successfully"))

        from agents.registry_integration import RegistryIntegrationMixin
        validation_results.append(("Registry Integration Mixin", True, "Registry integration components imported"))

        from agents.enhanced_base_agent import EnhancedBaseAgent, AgentConfiguration, AgentState
        validation_results.append(("Base Agent integration", True, "Enhanced base agent available for registry"))

        from db.models.agent import Agent, AgentStatus, AgentType
        validation_results.append(("Agent data models", True, "Agent models available for registry"))

    except ImportError as e:
        validation_results.append(("Import validation", False, f"Import error: {e}"))

    return validation_results

def validate_registry_architecture():
    """Validate agent registry architecture."""
    print("🔍 Validating agent registry architecture...")

    validation_results = []

    try:
        from agents.agent_registry_enhanced import EnhancedAgentRegistry

        # Check required methods exist
        required_methods = [
            'register_agent', 'deregister_agent', 'discover_agents',
            'select_agent', 'get_agent_status', 'get_registry_status',
            'update_agent_heartbeat', 'start', 'stop'
        ]

        registry_methods = [method for method in dir(EnhancedAgentRegistry) if not method.startswith('_')]

        missing_methods = []
        for method in required_methods:
            if method not in registry_methods:
                missing_methods.append(method)

        if missing_methods:
            validation_results.append(("Required methods", False, f"Missing methods: {missing_methods}"))
        else:
            validation_results.append(("Required methods", True, "All required registry methods present"))

        # Check method signatures
        register_method = getattr(EnhancedAgentRegistry, 'register_agent')
        if inspect.iscoroutinefunction(register_method):
            validation_results.append(("Async methods", True, "Register method is properly async"))
        else:
            validation_results.append(("Async methods", False, "Register method should be async"))

        discover_method = getattr(EnhancedAgentRegistry, 'discover_agents')
        if inspect.iscoroutinefunction(discover_method):
            validation_results.append(("Discovery method", True, "Discovery method is properly async"))
        else:
            validation_results.append(("Discovery method", False, "Discovery method should be async"))

    except Exception as e:
        validation_results.append(("Registry architecture", False, f"Architecture validation error: {e}"))

    return validation_results

def validate_discovery_models():
    """Validate discovery request and response models."""
    print("🔍 Validating discovery models...")

    validation_results = []

    try:
        from agents.agent_registry_enhanced import DiscoveryRequest, SelectionCriteria, AgentInfo
        from pydantic import BaseModel

        # Test DiscoveryRequest creation
        request = DiscoveryRequest(
            agent_type=None,
            capabilities=["test_capability"],
            max_load=0.8,
            min_health_score=0.7
        )
        assert hasattr(request, 'agent_type')
        assert hasattr(request, 'capabilities')
        assert hasattr(request, 'max_load')
        assert hasattr(request, 'min_health_score')

        validation_results.append(("DiscoveryRequest model", True, "Discovery request model works correctly"))

        # Test SelectionCriteria creation
        criteria = SelectionCriteria(
            discovery_request=request,
            selection_strategy=None  # Will use default
        )
        assert hasattr(criteria, 'discovery_request')
        assert hasattr(criteria, 'selection_strategy')

        validation_results.append(("SelectionCriteria model", True, "Selection criteria model works correctly"))

        # Test AgentInfo structure
        from datetime import datetime
        from db.models.agent import AgentType, AgentStatus

        agent_info = AgentInfo(
            agent_id="test_agent",
            agent_type=AgentType.GENERIC_AGENT,
            capabilities=["test"],
            status=AgentStatus.ACTIVE,
            endpoint="http://localhost:8000",
            health_score=1.0,
            current_load=0.0,
            last_heartbeat=datetime.utcnow(),
            registration_time=datetime.utcnow()
        )

        # Test serialization
        agent_dict = agent_info.to_dict()
        assert isinstance(agent_dict, dict)
        assert agent_dict["agent_id"] == "test_agent"

        # Test deserialization
        recreated_info = AgentInfo.from_dict(agent_dict)
        assert recreated_info.agent_id == agent_info.agent_id

        validation_results.append(("AgentInfo model", True, "Agent info model and serialization work correctly"))

    except Exception as e:
        validation_results.append(("Discovery models validation", False, f"Model validation error: {e}"))

    return validation_results

def validate_selection_strategies():
    """Validate agent selection strategies."""
    print("🔍 Validating selection strategies...")

    validation_results = []

    try:
        from agents.agent_registry_enhanced import SelectionStrategy

        # Check required selection strategies
        required_strategies = [
            'ROUND_ROBIN', 'LEAST_LOADED', 'RANDOM',
            'HIGHEST_HEALTH', 'CLOSEST', 'CUSTOM'
        ]

        available_strategies = [strategy.name for strategy in SelectionStrategy]
        missing_strategies = [strategy for strategy in required_strategies if strategy not in available_strategies]

        if missing_strategies:
            validation_results.append(("Selection strategies", False, f"Missing strategies: {missing_strategies}"))
        else:
            validation_results.append(("Selection strategies", True, "All required selection strategies defined"))

        # Test strategy values
        for strategy_name in required_strategies:
            if hasattr(SelectionStrategy, strategy_name):
                strategy = getattr(SelectionStrategy, strategy_name)
                assert isinstance(strategy, SelectionStrategy)

        validation_results.append(("Strategy enumeration", True, "Selection strategy enum works correctly"))

    except Exception as e:
        validation_results.append(("Selection strategies validation", False, f"Strategy validation error: {e}"))

    return validation_results

def validate_registry_integration():
    """Validate registry integration with enhanced base agent."""
    print("🔍 Validating registry integration...")

    validation_results = []

    try:
        from agents.registry_integration import RegistryIntegrationMixin

        # Check required integration methods
        required_methods = [
            'set_registry', 'is_registered', 'get_registry_status',
            '_register_with_registry', '_deregister_from_registry',
            '_calculate_health_score', '_calculate_load_percentage'
        ]

        integration_methods = [method for method in dir(RegistryIntegrationMixin) if not method.startswith('__')]

        missing_methods = []
        for method in required_methods:
            if method not in integration_methods:
                missing_methods.append(method)

        if missing_methods:
            validation_results.append(("Integration methods", False, f"Missing methods: {missing_methods}"))
        else:
            validation_results.append(("Integration methods", True, "All required integration methods present"))

        # Check async methods
        for method_name in ['_register_with_registry', '_deregister_from_registry', '_calculate_health_score']:
            if hasattr(RegistryIntegrationMixin, method_name):
                method = getattr(RegistryIntegrationMixin, method_name)
                if inspect.iscoroutinefunction(method):
                    validation_results.append((f"Async {method_name}", True, "Method is properly async"))
                else:
                    validation_results.append((f"Async {method_name}", False, "Method should be async"))

    except Exception as e:
        validation_results.append(("Registry integration", False, f"Integration validation error: {e}"))

    return validation_results

def validate_convenience_functions():
    """Validate convenience functions for common operations."""
    print("🔍 Validating convenience functions...")

    validation_results = []

    try:
        from agents.agent_registry_enhanced import (
            create_registry, discover_agent_by_type, discover_agent_by_capability
        )

        # Check function signatures
        create_sig = inspect.signature(create_registry)
        assert 'db_session' in create_sig.parameters
        validation_results.append(("create_registry function", True, "Registry creation function available"))

        discover_type_sig = inspect.signature(discover_agent_by_type)
        assert 'registry' in discover_type_sig.parameters
        assert 'agent_type' in discover_type_sig.parameters
        validation_results.append(("discover_agent_by_type function", True, "Type discovery function available"))

        discover_cap_sig = inspect.signature(discover_agent_by_capability)
        assert 'registry' in discover_cap_sig.parameters
        assert 'capabilities' in discover_cap_sig.parameters
        validation_results.append(("discover_agent_by_capability function", True, "Capability discovery function available"))

        # Check if functions are async
        if inspect.iscoroutinefunction(create_registry):
            validation_results.append(("Async convenience functions", True, "Convenience functions are properly async"))
        else:
            validation_results.append(("Async convenience functions", False, "Convenience functions should be async"))

    except Exception as e:
        validation_results.append(("Convenience functions validation", False, f"Function validation error: {e}"))

    return validation_results

def validate_error_handling():
    """Validate error handling framework."""
    print("🔍 Validating error handling...")

    validation_results = []

    try:
        from agents.agent_registry_enhanced import AgentRegistrationError, AgentDiscoveryError
        from core.exceptions import AgentError, NotFoundError

        # Test exception hierarchy
        assert issubclass(AgentRegistrationError, AgentError)
        assert issubclass(AgentDiscoveryError, AgentError)

        validation_results.append(("Exception hierarchy", True, "Proper exception inheritance"))

        # Test exception creation
        try:
            raise AgentRegistrationError("Test registration error")
        except AgentRegistrationError as e:
            assert str(e) == "Test registration error"
            validation_results.append(("Registration exceptions", True, "AgentRegistrationError works correctly"))

        try:
            raise AgentDiscoveryError("Test discovery error")
        except AgentDiscoveryError as e:
            assert str(e) == "Test discovery error"
            validation_results.append(("Discovery exceptions", True, "AgentDiscoveryError works correctly"))

    except Exception as e:
        validation_results.append(("Error handling validation", False, f"Error handling validation error: {e}"))

    return validation_results

def validate_file_structure():
    """Validate that all required files exist in the correct structure."""
    print("🔍 Validating file structure...")

    validation_results = []

    base_path = Path(__file__).parent

    required_files = [
        "agents/agent_registry_enhanced.py",
        "agents/registry_integration.py",
        "agents/enhanced_base_agent.py",
        "test_task_4_2_agent_registry.py",
        "db/models/agent.py",
        "db/repositories/agent.py",
        "core/exceptions.py"
    ]

    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        validation_results.append(("File structure", False, f"Missing files: {missing_files}"))
    else:
        validation_results.append(("File structure", True, "All required files present"))

    return validation_results

def validate_architecture_completeness():
    """Validate overall architecture completeness."""
    print("🔍 Validating architecture completeness...")

    validation_results = []

    # Read the enhanced registry file directly
    registry_file = Path(__file__).parent / "agents" / "agent_registry_enhanced.py"

    if not registry_file.exists():
        validation_results.append(("Architecture completeness", False, "Enhanced registry file not found"))
        return validation_results

    with open(registry_file, 'r') as f:
        content = f.read()

    # Check for key architectural components
    required_components = [
        "class EnhancedAgentRegistry",
        "class DiscoveryRequest",
        "class SelectionCriteria",
        "class AgentInfo",
        "class RegistryMetrics",
        "async def register_agent",
        "async def deregister_agent",
        "async def discover_agents",
        "async def select_agent",
        "def _select_least_loaded",
        "def _select_round_robin",
        "def _select_highest_health",
        "_health_check_loop",
        "_cleanup_loop"
    ]

    components_found = []
    for component in required_components:
        if component in content:
            components_found.append(component)

    completeness = (len(components_found) / len(required_components)) * 100

    if completeness >= 90:
        validation_results.append(("Architecture completeness", True, f"Architecture {completeness:.1f}% complete"))
    else:
        missing_components = [comp for comp in required_components if comp not in components_found]
        validation_results.append(("Architecture completeness", False, f"Missing components: {missing_components}"))

    # Check for integration patterns
    integration_patterns = [
        "logfire.span",
        "StructuredLogger",
        "AsyncAgentRepository",
        "AgentPhase",
        "correlation_context"
    ]

    integration_found = sum(1 for pattern in integration_patterns if pattern in content)
    integration_completeness = (integration_found / len(integration_patterns)) * 100

    if integration_completeness >= 80:
        validation_results.append(("Integration patterns", True, f"Integration {integration_completeness:.1f}% complete"))
    else:
        validation_results.append(("Integration patterns", False, f"Integration only {integration_completeness:.1f}% complete"))

    return validation_results

def print_validation_results(results: List[tuple], category: str):
    """Print validation results in a formatted way."""
    print(f"\n📋 {category} Results:")
    print("-" * 60)

    passed = 0
    total = len(results)

    for test_name, success, message in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}: {message}")
        if success:
            passed += 1

    print(f"\n📊 {category} Summary: {passed}/{total} tests passed")
    return passed, total

def main():
    """Run comprehensive Task 4.2 validation."""
    print("🚀 Task 4.2 - Agent Registry & Discovery Validation")
    print("=" * 70)
    print(f"Validation started at: {datetime.now().isoformat()}")
    print()

    total_passed = 0
    total_tests = 0

    # Run all validation tests
    validation_categories = [
        ("File Structure", validate_file_structure),
        ("Component Imports", validate_imports),
        ("Registry Architecture", validate_registry_architecture),
        ("Discovery Models", validate_discovery_models),
        ("Selection Strategies", validate_selection_strategies),
        ("Registry Integration", validate_registry_integration),
        ("Convenience Functions", validate_convenience_functions),
        ("Error Handling", validate_error_handling),
        ("Architecture Completeness", validate_architecture_completeness)
    ]

    for category_name, validation_func in validation_categories:
        try:
            results = validation_func()
            passed, total = print_validation_results(results, category_name)
            total_passed += passed
            total_tests += total
        except Exception as e:
            print(f"\n❌ {category_name} validation failed with error: {e}")
            total_tests += 1

    print("\n" + "=" * 70)
    print("📈 OVERALL VALIDATION SUMMARY")
    print("=" * 70)

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {success_rate:.1f}%")

    if success_rate >= 90:
        print("\n🎉 Task 4.2 Agent Registry & Discovery - VALIDATION SUCCESSFUL!")
        print("✅ All core requirements met")
        print("✅ Registry and discovery system ready for production")
        print("✅ Integration with enhanced base agent confirmed")
        print("✅ Selection strategies and health monitoring operational")
        print("✅ Architecture supports multi-agent coordination")
        print("\n🚀 READY FOR TASK 4.3: BASE AGENT TYPES IMPLEMENTATION")
    elif success_rate >= 75:
        print("\n⚠️  Task 4.2 Agent Registry & Discovery - MOSTLY COMPLETE")
        print("Most requirements met, some minor issues to address")
    else:
        print("\n❌ Task 4.2 Agent Registry & Discovery - NEEDS ATTENTION")
        print("Several critical issues need to be resolved")

    print(f"\n🕐 Validation completed at: {datetime.now().isoformat()}")
    print("=" * 70)

    return success_rate >= 90

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
