"""
Comprehensive Tests for Playbook Data Models

This module provides thorough testing of all playbook-related models including:
- Playbook entity operations and validations
- PlaybookStep management and dependencies
- PlaybookVariable typing and validation
- PlaybookExecution workflow and state tracking
- PlaybookStepExecution individual step tracking
- PlaybookTemplate reusability patterns

Test Coverage:
- Model creation and validation
- Relationship integrity
- Business logic and constraints
- State transitions and workflows
- Performance and edge cases
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from agentical.db.models.base import BaseModel
from agentical.db.models.playbook import (
    Playbook, PlaybookStep, PlaybookVariable, PlaybookExecution,
    PlaybookStepExecution, PlaybookTemplate,
    PlaybookCategory, PlaybookStatus, ExecutionStatus,
    StepType, StepStatus, VariableType
)
from agentical.db.models.user import User


class TestPlaybookModels:
    """Test suite for playbook models."""

    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        engine = create_engine("sqlite:///:memory:")
        BaseModel.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create test user
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="$2b$12$test_hash_for_testing"
        )
        session.add(user)
        session.commit()

        yield session
        session.close()

    @pytest.fixture
    def sample_playbook(self, db_session):
        """Create a sample playbook for testing."""
        playbook = Playbook(
            name="Test Incident Response",
            description="Test playbook for incident response",
            category=PlaybookCategory.INCIDENT_RESPONSE,
            created_by_user_id=1,
            tags=["test", "incident", "response"],
            success_criteria=["Issue resolved", "Documentation updated"]
        )
        db_session.add(playbook)
        db_session.commit()
        return playbook

    @pytest.fixture
    def sample_playbook_step(self, db_session, sample_playbook):
        """Create a sample playbook step."""
        step = PlaybookStep(
            playbook_id=sample_playbook.id,
            step_name="initial_assessment",
            display_name="Initial Assessment",
            description="Assess the incident severity",
            step_order=1,
            step_type=StepType.MANUAL,
            estimated_duration_minutes=15
        )
        db_session.add(step)
        db_session.commit()
        return step

    @pytest.fixture
    def sample_playbook_variable(self, db_session, sample_playbook):
        """Create a sample playbook variable."""
        variable = PlaybookVariable(
            playbook_id=sample_playbook.id,
            variable_name="incident_severity",
            display_name="Incident Severity",
            variable_type=VariableType.SELECT,
            default_value="medium",
            enum_values=["low", "medium", "high", "critical"]
        )
        db_session.add(variable)
        db_session.commit()
        return variable


class TestPlaybook:
    """Test Playbook model functionality."""

    def test_playbook_creation(self, db_session):
        """Test basic playbook creation."""
        playbook = Playbook(
            name="Test Playbook",
            description="A test playbook",
            category=PlaybookCategory.TESTING,
            created_by_user_id=1
        )

        db_session.add(playbook)
        db_session.commit()

        assert playbook.id is not None
        assert playbook.name == "Test Playbook"
        assert playbook.category == PlaybookCategory.TESTING
        assert playbook.status == PlaybookStatus.DRAFT
        assert playbook.is_executable is False  # No steps yet

    def test_playbook_validation(self, db_session):
        """Test playbook validation rules."""
        # Test invalid name
        with pytest.raises(ValueError):
            playbook = Playbook(
                name="",  # Empty name should fail
                category=PlaybookCategory.TESTING,
                created_by_user_id=1
            )
            db_session.add(playbook)
            db_session.commit()

    def test_playbook_tags_management(self, sample_playbook):
        """Test tag management functionality."""
        # Test getting tags
        tags = sample_playbook.get_tags()
        assert "test" in tags
        assert "incident" in tags

        # Test adding tags
        sample_playbook.add_tag("emergency")
        tags = sample_playbook.get_tags()
        assert "emergency" in tags

        # Test removing tags
        sample_playbook.remove_tag("test")
        tags = sample_playbook.get_tags()
        assert "test" not in tags

    def test_playbook_success_criteria(self, sample_playbook):
        """Test success criteria management."""
        criteria = sample_playbook.get_success_criteria()
        assert "Issue resolved" in criteria

        sample_playbook.add_success_criteria("Post-mortem completed")
        criteria = sample_playbook.get_success_criteria()
        assert "Post-mortem completed" in criteria

    def test_playbook_configuration(self, sample_playbook):
        """Test configuration management."""
        config = {
            "timeout_minutes": 60,
            "auto_escalate": True,
            "notification_channels": ["email", "slack"]
        }

        sample_playbook.set_configuration(config)
        retrieved_config = sample_playbook.get_configuration()

        assert retrieved_config["timeout_minutes"] == 60
        assert retrieved_config["auto_escalate"] is True

    def test_playbook_performance_metrics(self, sample_playbook):
        """Test performance metrics update."""
        sample_playbook.update_performance_metrics(
            execution_count=5,
            success_count=4,
            avg_duration=45.5
        )

        assert sample_playbook.success_rate == 0.8
        assert sample_playbook.total_executions == 5

    def test_playbook_status_transitions(self, sample_playbook):
        """Test playbook status transitions."""
        # Test publishing
        sample_playbook.publish()
        assert sample_playbook.status == PlaybookStatus.PUBLISHED
        assert sample_playbook.published_at is not None

        # Test archiving
        sample_playbook.archive()
        assert sample_playbook.status == PlaybookStatus.ARCHIVED

    def test_playbook_to_dict(self, sample_playbook):
        """Test dictionary conversion."""
        playbook_dict = sample_playbook.to_dict()

        assert playbook_dict["name"] == "Test Incident Response"
        assert playbook_dict["category"] == "incident_response"
        assert playbook_dict["status"] == "draft"
        assert "tags" in playbook_dict
        assert "success_criteria" in playbook_dict


class TestPlaybookStep:
    """Test PlaybookStep model functionality."""

    def test_step_creation(self, db_session, sample_playbook):
        """Test basic step creation."""
        step = PlaybookStep(
            playbook_id=sample_playbook.id,
            step_name="test_step",
            display_name="Test Step",
            step_order=1,
            step_type=StepType.AUTOMATED
        )

        db_session.add(step)
        db_session.commit()

        assert step.id is not None
        assert step.step_name == "test_step"
        assert step.step_type == StepType.AUTOMATED
        assert step.status == StepStatus.ACTIVE

    def test_step_dependencies(self, db_session, sample_playbook):
        """Test step dependency management."""
        step1 = PlaybookStep(
            playbook_id=sample_playbook.id,
            step_name="step1",
            step_order=1,
            step_type=StepType.MANUAL
        )
        step2 = PlaybookStep(
            playbook_id=sample_playbook.id,
            step_name="step2",
            step_order=2,
            step_type=StepType.MANUAL
        )

        db_session.add_all([step1, step2])
        db_session.commit()

        # Add dependency
        step2.add_dependency(step1.id)
        deps = step2.get_depends_on_steps()
        assert step1.id in deps

        # Remove dependency
        step2.remove_dependency(step1.id)
        deps = step2.get_depends_on_steps()
        assert step1.id not in deps

    def test_step_configuration(self, sample_playbook_step):
        """Test step configuration management."""
        config = {
            "timeout": 300,
            "retry_attempts": 3,
            "notifications": True
        }

        sample_playbook_step.set_configuration(config)
        retrieved = sample_playbook_step.get_configuration()

        assert retrieved["timeout"] == 300
        assert retrieved["retry_attempts"] == 3

    def test_step_performance_metrics(self, sample_playbook_step):
        """Test step performance tracking."""
        sample_playbook_step.update_performance_metrics(
            execution_count=10,
            success_count=8,
            avg_duration=120.5
        )

        assert sample_playbook_step.success_rate == 0.8


class TestPlaybookVariable:
    """Test PlaybookVariable model functionality."""

    def test_variable_creation(self, db_session, sample_playbook):
        """Test basic variable creation."""
        variable = PlaybookVariable(
            playbook_id=sample_playbook.id,
            variable_name="test_var",
            variable_type=VariableType.STRING,
            default_value="test_value"
        )

        db_session.add(variable)
        db_session.commit()

        assert variable.id is not None
        assert variable.variable_name == "test_var"
        assert variable.variable_type == VariableType.STRING

    def test_variable_validation(self, db_session, sample_playbook):
        """Test variable validation rules."""
        # Test invalid variable name
        with pytest.raises(ValueError):
            variable = PlaybookVariable(
                playbook_id=sample_playbook.id,
                variable_name="invalid-name!",  # Invalid characters
                variable_type=VariableType.STRING
            )
            db_session.add(variable)
            db_session.commit()

    def test_variable_value_validation(self, sample_playbook_variable):
        """Test variable value validation."""
        # Valid enum value
        assert sample_playbook_variable.validate_value("high") is True

        # Invalid enum value
        assert sample_playbook_variable.validate_value("invalid") is False

    def test_variable_value_operations(self, sample_playbook_variable):
        """Test variable value operations."""
        # Set value
        sample_playbook_variable.set_value("high")
        assert sample_playbook_variable.get_value() == "high"

        # Reset to default
        sample_playbook_variable.reset_to_default()
        assert sample_playbook_variable.get_value() == "medium"

    def test_variable_enum_management(self, sample_playbook_variable):
        """Test enum values management."""
        enum_values = sample_playbook_variable.get_enum_values()
        assert "low" in enum_values
        assert "critical" in enum_values


class TestPlaybookExecution:
    """Test PlaybookExecution model functionality."""

    @pytest.fixture
    def sample_execution(self, db_session, sample_playbook):
        """Create sample execution."""
        execution = PlaybookExecution(
            playbook_id=sample_playbook.id,
            executed_by_user_id=1,
            triggered_by="manual",
            input_variables={"severity": "high"}
        )
        db_session.add(execution)
        db_session.commit()
        return execution

    def test_execution_creation(self, sample_execution):
        """Test basic execution creation."""
        assert sample_execution.id is not None
        assert sample_execution.status == ExecutionStatus.PENDING
        assert sample_execution.triggered_by == "manual"

    def test_execution_lifecycle(self, sample_execution):
        """Test execution lifecycle management."""
        # Start execution
        sample_execution.start_execution({"severity": "high"})
        assert sample_execution.status == ExecutionStatus.RUNNING
        assert sample_execution.started_at is not None

        # Complete execution
        sample_execution.complete_execution(overall_success=True)
        assert sample_execution.status == ExecutionStatus.COMPLETED
        assert sample_execution.completed_at is not None
        assert sample_execution.overall_success is True

    def test_execution_pause_resume(self, sample_execution):
        """Test execution pause and resume."""
        sample_execution.start_execution()

        # Pause
        sample_execution.pause_execution()
        assert sample_execution.status == ExecutionStatus.PAUSED

        # Resume
        sample_execution.resume_execution()
        assert sample_execution.status == ExecutionStatus.RUNNING

    def test_execution_cancellation(self, sample_execution):
        """Test execution cancellation."""
        sample_execution.start_execution()
        sample_execution.cancel_execution("User requested cancellation")

        assert sample_execution.status == ExecutionStatus.CANCELLED
        assert "User requested" in sample_execution.cancellation_reason

    def test_execution_progress_tracking(self, sample_execution):
        """Test progress tracking."""
        sample_execution.update_progress(50, "Half way through")
        assert sample_execution.progress_percentage == 50.0

    def test_execution_success_criteria(self, sample_execution):
        """Test success criteria management."""
        sample_execution.mark_success_criteria("Issue resolved", True)
        criteria = sample_execution.get_success_criteria_met()
        assert criteria.get("Issue resolved") is True

    def test_execution_variables(self, sample_execution):
        """Test variable management."""
        # Input variables
        input_vars = sample_execution.get_input_variables()
        assert input_vars["severity"] == "high"

        # Set output variable
        sample_execution.set_variable("resolution_time", 45, is_output=True)
        output_vars = sample_execution.get_output_variables()
        assert output_vars["resolution_time"] == 45

    def test_execution_duration(self, sample_execution):
        """Test duration calculation."""
        sample_execution.start_execution()

        # Mock completion time
        with patch('agentical.db.models.playbook.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = sample_execution.started_at + timedelta(minutes=30)
            sample_execution.complete_execution()

        assert sample_execution.duration is not None
        assert sample_execution.duration.total_seconds() > 0


class TestPlaybookStepExecution:
    """Test PlaybookStepExecution model functionality."""

    @pytest.fixture
    def sample_step_execution(self, db_session, sample_execution, sample_playbook_step):
        """Create sample step execution."""
        step_execution = PlaybookStepExecution(
            playbook_execution_id=sample_execution.id,
            playbook_step_id=sample_playbook_step.id,
            step_order=1
        )
        db_session.add(step_execution)
        db_session.commit()
        return step_execution

    def test_step_execution_creation(self, sample_step_execution):
        """Test step execution creation."""
        assert sample_step_execution.id is not None
        assert sample_step_execution.status == StepStatus.PENDING
        assert sample_step_execution.step_order == 1

    def test_step_execution_lifecycle(self, sample_step_execution):
        """Test step execution lifecycle."""
        # Start step
        sample_step_execution.start_step()
        assert sample_step_execution.status == StepStatus.RUNNING
        assert sample_step_execution.started_at is not None

        # Complete step
        output_data = {"result": "success", "duration": 300}
        sample_step_execution.complete_step(output_data)
        assert sample_step_execution.status == StepStatus.COMPLETED
        assert sample_step_execution.output_data["result"] == "success"

    def test_step_execution_failure(self, sample_step_execution):
        """Test step execution failure."""
        sample_step_execution.start_step()
        sample_step_execution.fail_step("Connection timeout")

        assert sample_step_execution.status == StepStatus.FAILED
        assert sample_step_execution.error_message == "Connection timeout"

    def test_step_execution_skip(self, sample_step_execution):
        """Test step execution skip."""
        sample_step_execution.skip_step("Condition not met")

        assert sample_step_execution.status == StepStatus.SKIPPED
        assert "Skipped: Condition not met" in sample_step_execution.error_message

    def test_step_execution_duration(self, sample_step_execution):
        """Test step duration calculation."""
        sample_step_execution.start_step()

        # Mock completion
        with patch('agentical.db.models.playbook.datetime') as mock_datetime:
            start_time = sample_step_execution.started_at
            mock_datetime.utcnow.return_value = start_time + timedelta(minutes=5)
            sample_step_execution.complete_step()

        assert sample_step_execution.duration is not None
        assert sample_step_execution.execution_time_seconds == 300  # 5 minutes

    def test_step_execution_to_dict(self, sample_step_execution):
        """Test step execution dictionary conversion."""
        step_dict = sample_step_execution.to_dict()

        assert step_dict["step_order"] == 1
        assert step_dict["status"] == "pending"
        assert "playbook_execution_id" in step_dict


class TestPlaybookTemplate:
    """Test PlaybookTemplate model functionality."""

    @pytest.fixture
    def sample_template(self, db_session):
        """Create sample template."""
        template_data = {
            "description": "Standard incident response template",
            "steps": [
                {"name": "assess", "order": 1},
                {"name": "contain", "order": 2},
                {"name": "resolve", "order": 3}
            ],
            "variables": [
                {"name": "severity", "type": "select", "options": ["low", "high"]}
            ]
        }

        template = PlaybookTemplate(
            name="Incident Response Template",
            description="Template for incident response",
            category=PlaybookCategory.INCIDENT_RESPONSE,
            version="1.0.0",
            template_data=template_data,
            created_by="admin"
        )
        db_session.add(template)
        db_session.commit()
        return template

    def test_template_creation(self, sample_template):
        """Test template creation."""
        assert sample_template.id is not None
        assert sample_template.name == "Incident Response Template"
        assert sample_template.version == "1.0.0"
        assert sample_template.usage_count == 0

    def test_template_create_playbook(self, db_session, sample_template):
        """Test creating playbook from template."""
        playbook = sample_template.create_playbook(
            name="My Incident Response",
            created_by="user123"
        )

        assert playbook.name == "My Incident Response"
        assert playbook.category == PlaybookCategory.INCIDENT_RESPONSE
        assert playbook.created_by == "user123"
        assert sample_template.usage_count == 1

    def test_template_to_dict(self, sample_template):
        """Test template dictionary conversion."""
        template_dict = sample_template.to_dict()

        assert template_dict["name"] == "Incident Response Template"
        assert template_dict["category"] == "incident_response"
        assert template_dict["version"] == "1.0.0"
        assert "template_data" in template_dict


class TestPlaybookIntegration:
    """Test playbook model integration and relationships."""

    def test_playbook_with_steps_and_variables(self, db_session, sample_playbook):
        """Test complete playbook with steps and variables."""
        # Add steps
        step1 = PlaybookStep(
            playbook_id=sample_playbook.id,
            step_name="step1",
            step_order=1,
            step_type=StepType.MANUAL
        )
        step2 = PlaybookStep(
            playbook_id=sample_playbook.id,
            step_name="step2",
            step_order=2,
            step_type=StepType.AUTOMATED
        )

        # Add variables
        var1 = PlaybookVariable(
            playbook_id=sample_playbook.id,
            variable_name="priority",
            variable_type=VariableType.SELECT,
            enum_values=["low", "medium", "high"]
        )

        db_session.add_all([step1, step2, var1])
        db_session.commit()

        # Test relationships
        assert len(sample_playbook.steps) == 2
        assert len(sample_playbook.variables) == 1
        assert sample_playbook.step_count == 2
        assert sample_playbook.is_executable is True

    def test_execution_with_step_executions(self, db_session, sample_playbook):
        """Test execution with step executions."""
        # Create steps
        step1 = PlaybookStep(
            playbook_id=sample_playbook.id,
            step_name="step1",
            step_order=1,
            step_type=StepType.MANUAL
        )
        db_session.add(step1)
        db_session.commit()

        # Create execution
        execution = PlaybookExecution(
            playbook_id=sample_playbook.id,
            executed_by_user_id=1,
            triggered_by="test"
        )
        db_session.add(execution)
        db_session.commit()

        # Create step execution
        step_execution = PlaybookStepExecution(
            playbook_execution_id=execution.id,
            playbook_step_id=step1.id,
            step_order=1
        )
        db_session.add(step_execution)
        db_session.commit()

        # Test relationships
        assert len(execution.step_executions) == 1
        assert execution.step_executions[0].playbook_step_id == step1.id

    def test_cascade_delete(self, db_session, sample_playbook):
        """Test cascade delete functionality."""
        # Add related objects
        step = PlaybookStep(
            playbook_id=sample_playbook.id,
            step_name="test_step",
            step_order=1,
            step_type=StepType.MANUAL
        )
        variable = PlaybookVariable(
            playbook_id=sample_playbook.id,
            variable_name="test_var",
            variable_type=VariableType.STRING
        )
        execution = PlaybookExecution(
            playbook_id=sample_playbook.id,
            executed_by_user_id=1,
            triggered_by="test"
        )

        db_session.add_all([step, variable, execution])
        db_session.commit()

        step_id = step.id
        variable_id = variable.id
        execution_id = execution.id

        # Delete playbook
        db_session.delete(sample_playbook)
        db_session.commit()

        # Verify related objects are deleted
        assert db_session.query(PlaybookStep).filter_by(id=step_id).first() is None
        assert db_session.query(PlaybookVariable).filter_by(id=variable_id).first() is None
        assert db_session.query(PlaybookExecution).filter_by(id=execution_id).first() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
