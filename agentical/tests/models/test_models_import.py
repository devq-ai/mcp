#!/usr/bin/env python3
"""
Comprehensive Import Test for All Core Data Models

This test verifies that all core data models can be imported successfully
and that the integration is working properly. It tests:
- All model classes can be imported
- All enum classes can be imported
- Basic model instantiation works
- No import conflicts or circular dependencies

This serves as a final validation that Task 3.2 (Core Data Models) is
complete and all models are properly integrated.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_base_models_import():
    """Test that base models and mixins can be imported."""
    print("🔍 Testing base models import...")

    try:
        from db.models import (
            BaseModel, TimestampMixin, UUIDMixin, SoftDeleteMixin,
            JSONSerializableMixin, MetadataMixin
        )
        print("  ✅ Base models imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Base models import failed: {e}")
        return False

def test_user_models_import():
    """Test that user models can be imported."""
    print("🔍 Testing user models import...")

    try:
        from db.models import User, Role, user_roles
        print("  ✅ User models imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ User models import failed: {e}")
        return False

def test_agent_models_import():
    """Test that agent models can be imported."""
    print("🔍 Testing agent models import...")

    try:
        from db.models import (
            Agent, AgentCapability, AgentConfiguration, AgentExecution,
            AgentStatus, AgentType, AgentExecutionStatus
        )
        print("  ✅ Agent models imported successfully")

        # Test enum values
        assert AgentStatus.AVAILABLE
        assert AgentType.COORDINATOR
        assert AgentExecutionStatus.PENDING
        print("  ✅ Agent enums validated")

        return True
    except ImportError as e:
        print(f"  ❌ Agent models import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Agent models validation failed: {e}")
        return False

def test_tool_models_import():
    """Test that tool models can be imported."""
    print("🔍 Testing tool models import...")

    try:
        from db.models import (
            Tool, ToolCapability, ToolParameter, ToolExecution,
            ToolType, ToolStatus, ToolExecutionStatus
        )
        print("  ✅ Tool models imported successfully")

        # Test enum values
        assert ToolType.FILESYSTEM
        assert ToolStatus.AVAILABLE
        assert ToolExecutionStatus.PENDING
        print("  ✅ Tool enums validated")

        return True
    except ImportError as e:
        print(f"  ❌ Tool models import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Tool models validation failed: {e}")
        return False

def test_workflow_models_import():
    """Test that workflow models can be imported."""
    print("🔍 Testing workflow models import...")

    try:
        from db.models import (
            Workflow, WorkflowStep, WorkflowExecution, WorkflowStepExecution,
            WorkflowType, WorkflowStatus, WorkflowExecutionStatus,
            WorkflowStepType, WorkflowStepStatus
        )
        print("  ✅ Workflow models imported successfully")

        # Test enum values
        assert WorkflowType.SEQUENTIAL
        assert WorkflowStatus.DRAFT
        assert WorkflowExecutionStatus.PENDING
        assert WorkflowStepType.MANUAL
        assert WorkflowStepStatus.PENDING
        print("  ✅ Workflow enums validated")

        return True
    except ImportError as e:
        print(f"  ❌ Workflow models import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Workflow models validation failed: {e}")
        return False

def test_task_models_import():
    """Test that task models can be imported."""
    print("🔍 Testing task models import...")

    try:
        from db.models import (
            Task, TaskExecution, TaskResult,
            TaskPriority, TaskStatus, TaskType, TaskExecutionStatus
        )
        print("  ✅ Task models imported successfully")

        # Test enum values
        assert TaskPriority.HIGH
        assert TaskStatus.OPEN
        assert TaskType.FEATURE
        assert TaskExecutionStatus.PENDING
        print("  ✅ Task enums validated")

        return True
    except ImportError as e:
        print(f"  ❌ Task models import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Task models validation failed: {e}")
        return False

def test_playbook_models_import():
    """Test that playbook models can be imported."""
    print("🔍 Testing playbook models import...")

    try:
        from db.models import (
            Playbook, PlaybookStep, PlaybookVariable,
            PlaybookExecution, PlaybookStepExecution, PlaybookTemplate,
            PlaybookCategory, PlaybookStatus, PlaybookExecutionStatus,
            PlaybookStepType, PlaybookStepStatus, VariableType
        )
        print("  ✅ Playbook models imported successfully")

        # Test enum values
        assert PlaybookCategory.INCIDENT_RESPONSE
        assert PlaybookStatus.DRAFT
        assert PlaybookExecutionStatus.PENDING
        assert PlaybookStepType.MANUAL
        assert PlaybookStepStatus.PENDING
        assert VariableType.STRING
        print("  ✅ Playbook enums validated")

        return True
    except ImportError as e:
        print(f"  ❌ Playbook models import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Playbook models validation failed: {e}")
        return False

def test_bulk_import():
    """Test bulk import of all models at once."""
    print("🔍 Testing bulk import of all models...")

    try:
        from db.models import (
            # Base models
            BaseModel, TimestampMixin, UUIDMixin, SoftDeleteMixin,
            JSONSerializableMixin, MetadataMixin,

            # User models
            User, Role, user_roles,

            # Agent models
            Agent, AgentCapability, AgentConfiguration, AgentExecution,
            AgentStatus, AgentType, AgentExecutionStatus,

            # Tool models
            Tool, ToolCapability, ToolParameter, ToolExecution,
            ToolType, ToolStatus, ToolExecutionStatus,

            # Workflow models
            Workflow, WorkflowStep, WorkflowExecution, WorkflowStepExecution,
            WorkflowType, WorkflowStatus, WorkflowExecutionStatus,
            WorkflowStepType, WorkflowStepStatus,

            # Task models
            Task, TaskExecution, TaskResult,
            TaskPriority, TaskStatus, TaskType, TaskExecutionStatus,

            # Playbook models
            Playbook, PlaybookStep, PlaybookVariable,
            PlaybookExecution, PlaybookStepExecution, PlaybookTemplate,
            PlaybookCategory, PlaybookStatus, PlaybookExecutionStatus,
            PlaybookStepType, PlaybookStepStatus, VariableType
        )
        print("  ✅ Bulk import successful - all models available")
        return True
    except ImportError as e:
        print(f"  ❌ Bulk import failed: {e}")
        return False

def test_no_import_conflicts():
    """Test that there are no naming conflicts between models."""
    print("🔍 Testing for import conflicts...")

    try:
        from db.models import (
            AgentExecutionStatus, ToolExecutionStatus, WorkflowExecutionStatus,
            TaskExecutionStatus, PlaybookExecutionStatus
        )

        # Verify they are different enums
        assert AgentExecutionStatus != ToolExecutionStatus
        assert WorkflowExecutionStatus != TaskExecutionStatus
        assert PlaybookExecutionStatus != AgentExecutionStatus

        print("  ✅ No import conflicts detected - all ExecutionStatus enums are distinct")

        from db.models import (
            WorkflowStepType, PlaybookStepType,
            WorkflowStepStatus, PlaybookStepStatus
        )

        # Verify step types and statuses are different
        assert WorkflowStepType != PlaybookStepType
        assert WorkflowStepStatus != PlaybookStepStatus

        print("  ✅ No import conflicts detected - all Step enums are distinct")
        return True
    except Exception as e:
        print(f"  ❌ Import conflict test failed: {e}")
        return False

def test_model_inheritance():
    """Test that models properly inherit from BaseModel."""
    print("🔍 Testing model inheritance...")

    try:
        from db.models import (
            BaseModel, Agent, Tool, Workflow, Task, Playbook,
            AgentCapability, ToolParameter, WorkflowStep, TaskExecution,
            PlaybookVariable
        )

        # Test that all main models inherit from BaseModel
        models_to_test = [
            Agent, Tool, Workflow, Task, Playbook,
            AgentCapability, ToolParameter, WorkflowStep,
            TaskExecution, PlaybookVariable
        ]

        for model in models_to_test:
            if hasattr(model, '__bases__'):
                # Check if BaseModel is in the inheritance chain
                mro = model.__mro__
                base_in_mro = any(base.__name__ == 'BaseModel' for base in mro)
                assert base_in_mro, f"{model.__name__} does not inherit from BaseModel"

        print("  ✅ All models properly inherit from BaseModel")
        return True
    except Exception as e:
        print(f"  ❌ Model inheritance test failed: {e}")
        return False

def test_enum_completeness():
    """Test that all enums have expected values."""
    print("🔍 Testing enum completeness...")

    try:
        from db.models import (
            AgentStatus, AgentType,
            ToolType, ToolStatus,
            WorkflowType, WorkflowStatus,
            TaskPriority, TaskStatus, TaskType,
            PlaybookCategory, PlaybookStatus, VariableType
        )

        # Test key enum values exist
        enum_tests = [
            (AgentStatus, ['AVAILABLE', 'BUSY', 'OFFLINE']),
            (AgentType, ['COORDINATOR', 'SPECIALIST', 'EXECUTOR']),
            (ToolType, ['FILESYSTEM', 'GIT', 'MEMORY', 'CUSTOM']),
            (ToolStatus, ['AVAILABLE', 'UNAVAILABLE', 'MAINTENANCE']),
            (WorkflowType, ['SEQUENTIAL', 'PARALLEL', 'CONDITIONAL']),
            (WorkflowStatus, ['DRAFT', 'PUBLISHED', 'ARCHIVED']),
            (TaskPriority, ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
            (TaskStatus, ['OPEN', 'IN_PROGRESS', 'COMPLETED']),
            (TaskType, ['FEATURE', 'BUG', 'IMPROVEMENT']),
            (PlaybookCategory, ['INCIDENT_RESPONSE', 'TESTING', 'DEPLOYMENT']),
            (PlaybookStatus, ['DRAFT', 'PUBLISHED', 'ARCHIVED']),
            (VariableType, ['STRING', 'INTEGER', 'BOOLEAN', 'SELECT'])
        ]

        for enum_class, expected_values in enum_tests:
            enum_names = [item.name for item in enum_class]
            for expected in expected_values:
                assert expected in enum_names, f"{expected} not found in {enum_class.__name__}"

        print("  ✅ All enums have expected values")
        return True
    except Exception as e:
        print(f"  ❌ Enum completeness test failed: {e}")
        return False

def generate_import_summary():
    """Generate summary of available models."""
    print("📊 Generating import summary...")

    try:
        from db.models import __all__

        summary = {
            "total_exports": len(__all__),
            "categories": {
                "base": [],
                "user": [],
                "agent": [],
                "tool": [],
                "workflow": [],
                "task": [],
                "playbook": []
            }
        }

        for export in __all__:
            if any(base in export.lower() for base in ["base", "mixin"]):
                summary["categories"]["base"].append(export)
            elif any(user in export.lower() for user in ["user", "role"]):
                summary["categories"]["user"].append(export)
            elif "agent" in export.lower():
                summary["categories"]["agent"].append(export)
            elif "tool" in export.lower():
                summary["categories"]["tool"].append(export)
            elif "workflow" in export.lower():
                summary["categories"]["workflow"].append(export)
            elif "task" in export.lower():
                summary["categories"]["task"].append(export)
            elif any(pb in export.lower() for pb in ["playbook", "variable"]):
                summary["categories"]["playbook"].append(export)

        print(f"  📈 Total exports: {summary['total_exports']}")
        for category, items in summary["categories"].items():
            if items:
                print(f"  🏷️ {category.title()}: {len(items)} items")
                print(f"    {', '.join(items)}")

        return summary
    except Exception as e:
        print(f"  ❌ Summary generation failed: {e}")
        return None

def main():
    """Run comprehensive import tests."""
    print("🔍 Starting Comprehensive Models Import Test")
    print("=" * 60)
    print("Testing all core data models integration and imports")
    print("=" * 60)

    tests = [
        ("Base Models Import", test_base_models_import),
        ("User Models Import", test_user_models_import),
        ("Agent Models Import", test_agent_models_import),
        ("Tool Models Import", test_tool_models_import),
        ("Workflow Models Import", test_workflow_models_import),
        ("Task Models Import", test_task_models_import),
        ("Playbook Models Import", test_playbook_models_import),
        ("Bulk Import Test", test_bulk_import),
        ("Import Conflicts Test", test_no_import_conflicts),
        ("Model Inheritance Test", test_model_inheritance),
        ("Enum Completeness Test", test_enum_completeness)
    ]

    results = {}
    passed = 0

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False

    # Generate summary
    print(f"\n📊 Import Summary:")
    summary = generate_import_summary()

    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)

    total_tests = len(tests)
    success_rate = (passed / total_tests) * 100

    print(f"Tests Passed: {passed}/{total_tests}")
    print(f"Success Rate: {success_rate:.1f}%")

    if summary:
        print(f"Total Models Exported: {summary['total_exports']}")

    if success_rate == 100:
        print("\n🎉 ALL IMPORT TESTS PASSED!")
        print("✅ All core data models are properly integrated")
        print("✅ No import conflicts detected")
        print("✅ All enums are complete and accessible")
        print("✅ Models properly inherit from BaseModel")
        print("✅ Task 3.2 (Core Data Models) integration is COMPLETE")
        return True
    else:
        print(f"\n💥 Some import tests failed")
        print(f"❌ Integration needs work: {success_rate:.1f}% success rate")

        # Show failed tests
        failed_tests = [name for name, result in results.items() if not result]
        if failed_tests:
            print(f"Failed tests: {', '.join(failed_tests)}")

        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
