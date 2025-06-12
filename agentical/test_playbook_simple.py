"""
Simple Validation Test for Playbook Models

This is a lightweight test to verify that the playbook models can be imported
and basic instances can be created without errors.
"""

import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

def test_model_imports():
    """Test that all playbook models can be imported successfully."""
    try:
        from agentical.db.models.playbook import (
            Playbook, PlaybookStep, PlaybookVariable, PlaybookExecution,
            PlaybookStepExecution, PlaybookTemplate,
            PlaybookCategory, PlaybookStatus, ExecutionStatus,
            StepType, StepStatus, VariableType
        )
        print("✅ All playbook models imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_enum_values():
    """Test that all enums have expected values."""
    try:
        from agentical.db.models.playbook import (
            PlaybookCategory, PlaybookStatus, ExecutionStatus,
            StepType, StepStatus, VariableType
        )

        # Test PlaybookCategory
        assert PlaybookCategory.INCIDENT_RESPONSE == "incident_response"
        assert PlaybookCategory.TESTING == "testing"
        print("✅ PlaybookCategory enum values correct")

        # Test PlaybookStatus
        assert PlaybookStatus.DRAFT == "draft"
        assert PlaybookStatus.PUBLISHED == "published"
        print("✅ PlaybookStatus enum values correct")

        # Test ExecutionStatus
        assert ExecutionStatus.PENDING == "pending"
        assert ExecutionStatus.RUNNING == "running"
        assert ExecutionStatus.COMPLETED == "completed"
        print("✅ ExecutionStatus enum values correct")

        # Test StepType
        assert StepType.MANUAL == "manual"
        assert StepType.AUTOMATED == "automated"
        print("✅ StepType enum values correct")

        # Test StepStatus
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.COMPLETED == "completed"
        print("✅ StepStatus enum values correct")

        # Test VariableType
        assert VariableType.STRING == "string"
        assert VariableType.INTEGER == "integer"
        assert VariableType.SELECT == "select"
        print("✅ VariableType enum values correct")

        return True
    except Exception as e:
        print(f"❌ Enum test error: {e}")
        return False

def test_model_creation():
    """Test basic model instantiation without database."""
    try:
        from agentical.db.models.playbook import (
            Playbook, PlaybookStep, PlaybookVariable, PlaybookExecution,
            PlaybookStepExecution, PlaybookTemplate,
            PlaybookCategory, PlaybookStatus, ExecutionStatus,
            StepType, StepStatus, VariableType
        )

        # Test Playbook creation
        playbook = Playbook(
            name="Test Playbook",
            description="A test playbook",
            category=PlaybookCategory.TESTING,
            created_by_user_id=1
        )
        assert playbook.name == "Test Playbook"
        assert playbook.category == PlaybookCategory.TESTING
        print("✅ Playbook model creation successful")

        # Test PlaybookStep creation
        step = PlaybookStep(
            playbook_id=1,
            step_name="test_step",
            display_name="Test Step",
            step_order=1,
            step_type=StepType.MANUAL
        )
        assert step.step_name == "test_step"
        assert step.step_type == StepType.MANUAL
        print("✅ PlaybookStep model creation successful")

        # Test PlaybookVariable creation
        variable = PlaybookVariable(
            playbook_id=1,
            variable_name="test_var",
            variable_type=VariableType.STRING,
            default_value="test"
        )
        assert variable.variable_name == "test_var"
        assert variable.variable_type == VariableType.STRING
        print("✅ PlaybookVariable model creation successful")

        # Test PlaybookExecution creation
        execution = PlaybookExecution(
            playbook_id=1,
            executed_by_user_id=1,
            triggered_by="test"
        )
        assert execution.triggered_by == "test"
        assert execution.status == ExecutionStatus.PENDING
        print("✅ PlaybookExecution model creation successful")

        # Test PlaybookStepExecution creation
        step_execution = PlaybookStepExecution(
            playbook_execution_id="test-exec-id",
            playbook_step_id="test-step-id",
            step_order=1
        )
        assert step_execution.step_order == 1
        assert step_execution.status == StepStatus.PENDING
        print("✅ PlaybookStepExecution model creation successful")

        # Test PlaybookTemplate creation
        template = PlaybookTemplate(
            name="Test Template",
            description="A test template",
            category=PlaybookCategory.TESTING,
            version="1.0.0",
            template_data={"test": "data"},
            created_by="admin"
        )
        assert template.name == "Test Template"
        assert template.version == "1.0.0"
        print("✅ PlaybookTemplate model creation successful")

        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_methods():
    """Test model methods that don't require database access."""
    try:
        from agentical.db.models.playbook import (
            Playbook, PlaybookVariable, PlaybookCategory, VariableType
        )

        # Test Playbook methods
        playbook = Playbook(
            name="Test Playbook",
            category=PlaybookCategory.TESTING,
            created_by_user_id=1,
            tags=["test", "validation"]
        )

        # Test tag management
        tags = playbook.get_tags()
        assert "test" in tags
        assert "validation" in tags
        print("✅ Playbook tag management works")

        playbook.add_tag("new_tag")
        tags = playbook.get_tags()
        assert "new_tag" in tags
        print("✅ Playbook add_tag works")

        playbook.remove_tag("test")
        tags = playbook.get_tags()
        assert "test" not in tags
        print("✅ Playbook remove_tag works")

        # Test PlaybookVariable validation
        variable = PlaybookVariable(
            playbook_id=1,
            variable_name="test_var",
            variable_type=VariableType.SELECT,
            enum_values=["option1", "option2", "option3"],
            default_value="option1"
        )

        # Test enum validation
        assert variable.validate_value("option1") is True
        assert variable.validate_value("invalid") is False
        print("✅ PlaybookVariable enum validation works")

        # Test value operations
        variable.set_value("option2")
        assert variable.get_value() == "option2"
        print("✅ PlaybookVariable value operations work")

        variable.reset_to_default()
        assert variable.get_value() == "option1"
        print("✅ PlaybookVariable reset_to_default works")

        return True
    except Exception as e:
        print(f"❌ Model methods error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🔍 Starting Playbook Models Validation")
    print("=" * 50)

    tests = [
        ("Model Imports", test_model_imports),
        ("Enum Values", test_enum_values),
        ("Model Creation", test_model_creation),
        ("Model Methods", test_model_methods)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All playbook model validations passed!")
        return True
    else:
        print("💥 Some validations failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
