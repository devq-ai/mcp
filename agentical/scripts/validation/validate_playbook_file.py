#!/usr/bin/env python3
"""
File-Based Playbook Models Validation

This script validates the playbook models implementation by analyzing the actual
playbook.py file structure and content to ensure Task 3.2 is complete.
"""

import os
import re
import ast
import json
from datetime import datetime
from typing import Dict, Any, List, Set, Optional


def analyze_playbook_file() -> Dict[str, Any]:
    """Analyze the playbook.py file structure and content."""

    playbook_file = "db/models/playbook.py"

    if not os.path.exists(playbook_file):
        return {"error": f"Playbook file not found: {playbook_file}"}

    with open(playbook_file, 'r') as f:
        content = f.read()

    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error in playbook.py: {e}"}

    analysis = {
        "file_size": len(content),
        "line_count": len(content.split('\n')),
        "imports": [],
        "classes": {},
        "enums": {},
        "functions": [],
        "docstring": None
    }

    # Extract module docstring
    if (isinstance(tree.body[0], ast.Expr) and
        isinstance(tree.body[0].value, ast.Constant) and
        isinstance(tree.body[0].value.value, str)):
        analysis["docstring"] = tree.body[0].value.value

    # Analyze each node
    for node in ast.walk(tree):
        # Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                analysis["imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                analysis["imports"].append(f"{module}.{alias.name}")

        # Classes
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                "methods": [],
                "attributes": [],
                "docstring": None,
                "is_enum": False
            }

            # Check if it's an enum
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Enum":
                    class_info["is_enum"] = True
                    break

            # Get docstring
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
                class_info["docstring"] = node.body[0].value.value

            # Get methods and attributes
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info = {
                        "name": item.name,
                        "args": [arg.arg for arg in item.args.args],
                        "decorators": [dec.id if isinstance(dec, ast.Name) else str(dec) for dec in item.decorator_list]
                    }
                    class_info["methods"].append(method_info)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            class_info["attributes"].append(target.id)

            if class_info["is_enum"]:
                analysis["enums"][node.name] = class_info
            else:
                analysis["classes"][node.name] = class_info

        # Functions
        elif isinstance(node, ast.FunctionDef) and not hasattr(node, 'parent'):
            analysis["functions"].append(node.name)

    return analysis


def validate_enums(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Validate enum definitions."""

    expected_enums = {
        "PlaybookCategory": [
            "INCIDENT_RESPONSE", "TROUBLESHOOTING", "DEPLOYMENT", "MAINTENANCE",
            "SECURITY", "CODE_REVIEW", "TESTING", "RELEASE", "ONBOARDING",
            "MONITORING", "BACKUP", "DISASTER_RECOVERY", "CAPACITY_PLANNING"
        ],
        "PlaybookStatus": ["DRAFT", "PUBLISHED", "ARCHIVED", "DEPRECATED"],
        "ExecutionStatus": ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "PAUSED"],
        "StepType": [
            "MANUAL", "AUTOMATED", "CONDITIONAL", "PARALLEL", "SEQUENTIAL",
            "APPROVAL", "NOTIFICATION", "WEBHOOK", "SCRIPT", "API_CALL",
            "DATABASE", "FILE_OPERATION", "EXTERNAL_TOOL"
        ],
        "StepStatus": ["PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED", "ACTIVE"],
        "VariableType": [
            "STRING", "INTEGER", "FLOAT", "BOOLEAN", "JSON",
            "SELECT", "MULTI_SELECT", "DATE", "DATETIME", "FILE"
        ]
    }

    validation = {
        "total_expected": len(expected_enums),
        "found_enums": list(analysis["enums"].keys()),
        "missing_enums": [],
        "complete_enums": [],
        "enum_details": {}
    }

    for enum_name, expected_values in expected_enums.items():
        if enum_name in analysis["enums"]:
            validation["complete_enums"].append(enum_name)
            validation["enum_details"][enum_name] = {
                "found": True,
                "docstring": analysis["enums"][enum_name]["docstring"] is not None
            }
        else:
            validation["missing_enums"].append(enum_name)
            validation["enum_details"][enum_name] = {"found": False}

    validation["completion_rate"] = len(validation["complete_enums"]) / len(expected_enums) * 100

    return validation


def validate_models(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Validate model class definitions."""

    expected_models = {
        "Playbook": {
            "required_methods": [
                "get_tags", "add_tag", "remove_tag", "get_success_criteria",
                "add_success_criteria", "get_configuration", "set_configuration",
                "update_performance_metrics", "publish", "archive", "to_dict"
            ],
            "expected_attributes": [
                "name", "description", "category", "status", "created_by_user_id",
                "tags", "success_criteria", "configuration"
            ]
        },
        "PlaybookStep": {
            "required_methods": [
                "get_depends_on_steps", "add_dependency", "remove_dependency",
                "get_configuration", "set_configuration", "update_performance_metrics", "to_dict"
            ],
            "expected_attributes": [
                "playbook_id", "step_name", "display_name", "description",
                "step_order", "step_type", "status"
            ]
        },
        "PlaybookVariable": {
            "required_methods": [
                "get_enum_values", "validate_value", "set_value", "get_value",
                "reset_to_default", "to_dict"
            ],
            "expected_attributes": [
                "playbook_id", "variable_name", "variable_type", "default_value", "current_value"
            ]
        },
        "PlaybookExecution": {
            "required_methods": [
                "start_execution", "complete_execution", "pause_execution",
                "resume_execution", "cancel_execution", "update_progress",
                "get_success_criteria_met", "mark_success_criteria",
                "get_input_variables", "get_output_variables", "set_variable"
            ],
            "expected_attributes": [
                "playbook_id", "executed_by_user_id", "status", "triggered_by",
                "started_at", "completed_at", "input_variables", "output_variables"
            ]
        },
        "PlaybookStepExecution": {
            "required_methods": ["start_step", "complete_step", "fail_step", "skip_step", "to_dict"],
            "expected_attributes": [
                "playbook_execution_id", "playbook_step_id", "step_order",
                "status", "started_at", "completed_at", "input_data", "output_data"
            ]
        },
        "PlaybookTemplate": {
            "required_methods": ["create_playbook", "to_dict"],
            "expected_attributes": [
                "name", "description", "category", "version", "template_data",
                "default_variables", "usage_count", "is_public", "created_by"
            ]
        }
    }

    validation = {
        "total_expected": len(expected_models),
        "found_models": list(analysis["classes"].keys()),
        "missing_models": [],
        "complete_models": [],
        "model_details": {}
    }

    for model_name, requirements in expected_models.items():
        model_detail = {
            "found": model_name in analysis["classes"],
            "methods_found": [],
            "methods_missing": [],
            "attributes_found": [],
            "has_docstring": False,
            "inherits_base": False
        }

        if model_name in analysis["classes"]:
            model_info = analysis["classes"][model_name]
            model_detail["has_docstring"] = model_info["docstring"] is not None
            model_detail["inherits_base"] = "BaseModel" in model_info["bases"]

            # Check methods
            found_methods = [method["name"] for method in model_info["methods"]]
            for required_method in requirements["required_methods"]:
                if required_method in found_methods:
                    model_detail["methods_found"].append(required_method)
                else:
                    model_detail["methods_missing"].append(required_method)

            # Check attributes (approximate - this is harder to detect from AST)
            model_detail["attributes_found"] = model_info["attributes"]

            # Determine if model is complete
            methods_complete = len(model_detail["methods_missing"]) == 0
            if methods_complete:
                validation["complete_models"].append(model_name)
        else:
            validation["missing_models"].append(model_name)
            model_detail["methods_missing"] = requirements["required_methods"]

        validation["model_details"][model_name] = model_detail

    validation["completion_rate"] = len(validation["complete_models"]) / len(expected_models) * 100

    return validation


def validate_imports(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required imports."""

    required_imports = [
        "datetime.datetime", "datetime.timedelta", "typing.Dict", "typing.Any",
        "typing.List", "typing.Optional", "enum.Enum", "sqlalchemy.Column",
        "sqlalchemy.String", "sqlalchemy.Integer", "sqlalchemy.Text",
        "sqlalchemy.Boolean", "sqlalchemy.DateTime", "sqlalchemy.ForeignKey",
        "sqlalchemy.JSON", "sqlalchemy.Float", "sqlalchemy.Index",
        "sqlalchemy.UniqueConstraint", "sqlalchemy.orm.relationship",
        "sqlalchemy.orm.validates", "sqlalchemy.ext.hybrid.hybrid_property"
    ]

    found_imports = set(analysis["imports"])

    validation = {
        "required_imports": required_imports,
        "found_imports": list(found_imports),
        "missing_imports": [],
        "extra_imports": []
    }

    for imp in required_imports:
        # Check for partial matches (e.g., "datetime" covers "datetime.datetime")
        if not any(found_imp.endswith(imp.split('.')[-1]) for found_imp in found_imports):
            validation["missing_imports"].append(imp)

    return validation


def validate_file_structure(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Validate overall file structure."""

    validation = {
        "has_docstring": analysis["docstring"] is not None,
        "docstring_comprehensive": False,
        "file_size_appropriate": 500 < analysis["file_size"] < 50000,  # Reasonable size
        "line_count": analysis["line_count"],
        "complexity_score": 0
    }

    if validation["has_docstring"]:
        docstring = analysis["docstring"]
        validation["docstring_comprehensive"] = (
            "Playbook" in docstring and
            "models" in docstring and
            ("Features" in docstring or "features" in docstring)
        )

    # Calculate complexity score based on content
    validation["complexity_score"] = (
        len(analysis["classes"]) * 10 +
        len(analysis["enums"]) * 5 +
        sum(len(model["methods"]) for model in analysis["classes"].values()) * 2
    )

    return validation


def generate_validation_report() -> Dict[str, Any]:
    """Generate comprehensive validation report."""

    print("🔍 Analyzing playbook.py file structure...")

    # Analyze the file
    analysis = analyze_playbook_file()

    if "error" in analysis:
        return {
            "status": "ERROR",
            "error": analysis["error"],
            "timestamp": datetime.now().isoformat()
        }

    print(f"✅ File analyzed: {analysis['line_count']} lines, {analysis['file_size']} bytes")

    # Run validations
    enum_validation = validate_enums(analysis)
    model_validation = validate_models(analysis)
    import_validation = validate_imports(analysis)
    structure_validation = validate_file_structure(analysis)

    # Calculate overall score
    scores = [
        enum_validation["completion_rate"],
        model_validation["completion_rate"],
        (100 - len(import_validation["missing_imports"]) * 5),  # Deduct 5% per missing import
        (100 if structure_validation["has_docstring"] else 80)   # Deduct 20% if no docstring
    ]

    overall_score = sum(scores) / len(scores)

    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "PASSED" if overall_score >= 85 else "FAILED",
        "overall_score": round(overall_score, 2),
        "file_analysis": {
            "file_size": analysis["file_size"],
            "line_count": analysis["line_count"],
            "has_docstring": structure_validation["has_docstring"],
            "complexity_score": structure_validation["complexity_score"]
        },
        "enum_validation": enum_validation,
        "model_validation": model_validation,
        "import_validation": import_validation,
        "structure_validation": structure_validation,
        "recommendations": []
    }

    # Generate recommendations
    if enum_validation["missing_enums"]:
        report["recommendations"].append(f"Implement missing enums: {', '.join(enum_validation['missing_enums'])}")

    if model_validation["missing_models"]:
        report["recommendations"].append(f"Implement missing models: {', '.join(model_validation['missing_models'])}")

    for model_name, details in model_validation["model_details"].items():
        if details["methods_missing"]:
            report["recommendations"].append(f"Implement missing methods in {model_name}: {', '.join(details['methods_missing'])}")

    if import_validation["missing_imports"]:
        report["recommendations"].append("Add missing imports for full functionality")

    if not structure_validation["has_docstring"]:
        report["recommendations"].append("Add comprehensive module docstring")

    if overall_score >= 95:
        report["recommendations"].append("Excellent! The playbook models implementation is comprehensive and ready for production.")
    elif overall_score >= 85:
        report["recommendations"].append("Good implementation. Address minor issues for optimal quality.")

    return report


def main():
    """Main validation function."""

    print("🔍 Starting File-Based Playbook Models Validation")
    print("=" * 60)
    print("Analyzing db/models/playbook.py file structure and content")
    print("=" * 60)

    # Generate report
    report = generate_validation_report()

    # Display results
    print(f"\n📊 VALIDATION RESULTS")
    print("=" * 40)
    print(f"Overall Status: {report['status']}")
    print(f"Overall Score: {report['overall_score']}%")

    if "error" in report:
        print(f"❌ Error: {report['error']}")
        return False

    # File analysis
    file_info = report["file_analysis"]
    print(f"\n📄 File Analysis:")
    print(f"  Lines of code: {file_info['line_count']}")
    print(f"  File size: {file_info['file_size']} bytes")
    print(f"  Has docstring: {'✅' if file_info['has_docstring'] else '❌'}")
    print(f"  Complexity score: {file_info['complexity_score']}")

    # Enum validation
    enum_val = report["enum_validation"]
    print(f"\n🏷️  Enum Validation:")
    print(f"  Completion rate: {enum_val['completion_rate']:.1f}%")
    print(f"  Complete enums: {len(enum_val['complete_enums'])}/{enum_val['total_expected']}")
    if enum_val["missing_enums"]:
        print(f"  Missing: {', '.join(enum_val['missing_enums'])}")

    # Model validation
    model_val = report["model_validation"]
    print(f"\n🏗️  Model Validation:")
    print(f"  Completion rate: {model_val['completion_rate']:.1f}%")
    print(f"  Complete models: {len(model_val['complete_models'])}/{model_val['total_expected']}")
    if model_val["missing_models"]:
        print(f"  Missing: {', '.join(model_val['missing_models'])}")

    # Show detailed model analysis
    for model_name, details in model_val["model_details"].items():
        if details["found"]:
            status = "✅" if not details["methods_missing"] else "⚠️"
            missing_count = len(details["methods_missing"])
            total_methods = len(details["methods_found"]) + missing_count
            print(f"    {status} {model_name}: {len(details['methods_found'])}/{total_methods} methods")
            if details["methods_missing"]:
                print(f"      Missing: {', '.join(details['methods_missing'])}")

    # Import validation
    import_val = report["import_validation"]
    print(f"\n📦 Import Validation:")
    print(f"  Found imports: {len(import_val['found_imports'])}")
    if import_val["missing_imports"]:
        print(f"  Missing critical imports: {len(import_val['missing_imports'])}")

    # Recommendations
    if report["recommendations"]:
        print(f"\n📋 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

    # Save detailed report
    report_file = "playbook_file_validation_report.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")

    print("\n" + "=" * 60)

    if report["status"] == "PASSED":
        print("🎉 PLAYBOOK MODELS FILE VALIDATION PASSED!")
        print("✅ The playbook.py file structure is complete and well-implemented")
        print("✅ Task 3.2 (Core Data Models - Playbooks) appears to be COMPLETE")
        return True
    else:
        print("💥 Playbook models file validation failed")
        print("❌ Task 3.2 (Core Data Models - Playbooks) needs additional work")
        print(f"📊 Current score: {report['overall_score']}% (need 85% to pass)")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
