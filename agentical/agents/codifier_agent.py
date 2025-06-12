"""
Codifier Agent Implementation for Agentical Framework

This module provides the CodifierAgent implementation for documentation,
logging, knowledge codification, and information structuring tasks.

Features:
- Documentation generation and maintenance
- Structured logging and data organization
- Knowledge base management
- Code documentation and comments
- Process documentation and runbooks
- Information architecture and taxonomy
- Content standardization and formatting
"""

from typing import Dict, Any, List, Optional, Set, Union, Tuple
from datetime import datetime
import asyncio
import json
import re
from pathlib import Path
from enum import Enum

import logfire
from pydantic import BaseModel, Field, validator

from agentical.agents.enhanced_base_agent import EnhancedBaseAgent
from agentical.db.models.agent import AgentType, AgentStatus
from agentical.core.exceptions import AgentExecutionError, ValidationError
from agentical.core.structured_logging import StructuredLogger, OperationType, AgentPhase


class DocumentationType(Enum):
    """Types of documentation that can be generated."""
    API_DOCUMENTATION = "api_documentation"
    CODE_DOCUMENTATION = "code_documentation"
    USER_MANUAL = "user_manual"
    TECHNICAL_SPECIFICATION = "technical_specification"
    PROCESS_DOCUMENTATION = "process_documentation"
    RUNBOOK = "runbook"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    CHANGELOG = "changelog"
    README = "readme"
    ARCHITECTURE_DOCUMENT = "architecture_document"


class LogFormat(Enum):
    """Supported logging formats."""
    JSON = "json"
    STRUCTURED = "structured"
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    CSV = "csv"
    XML = "xml"


class DocumentationRequest(BaseModel):
    """Request model for documentation generation tasks."""
    documentation_type: DocumentationType = Field(..., description="Type of documentation to generate")
    source_path: Optional[str] = Field(default=None, description="Path to source code or files")
    content: Optional[str] = Field(default=None, description="Raw content to document")
    output_format: str = Field(default="markdown", description="Output format (markdown, html, pdf, etc.)")
    template: Optional[str] = Field(default=None, description="Documentation template to use")
    include_examples: bool = Field(default=True, description="Include code examples")
    include_diagrams: bool = Field(default=False, description="Include architectural diagrams")
    target_audience: str = Field(default="developer", description="Target audience level")
    language: str = Field(default="en", description="Documentation language")


class LogStructuringRequest(BaseModel):
    """Request model for log structuring and analysis tasks."""
    log_data: Union[str, List[str], Dict[str, Any]] = Field(..., description="Log data to structure")
    source_format: LogFormat = Field(..., description="Source log format")
    target_format: LogFormat = Field(..., description="Target output format")
    parse_patterns: Optional[List[str]] = Field(default=None, description="Custom parsing patterns")
    extract_fields: Optional[List[str]] = Field(default=None, description="Specific fields to extract")
    filter_criteria: Optional[Dict[str, Any]] = Field(default=None, description="Filtering criteria")
    aggregate_data: bool = Field(default=False, description="Perform data aggregation")
    time_range: Optional[Tuple[datetime, datetime]] = Field(default=None, description="Time range filter")


class KnowledgeCodeRequest(BaseModel):
    """Request model for knowledge codification tasks."""
    knowledge_source: Union[str, List[str]] = Field(..., description="Source of knowledge to codify")
    knowledge_type: str = Field(..., description="Type of knowledge (process, technical, business)")
    output_structure: str = Field(default="structured", description="Output structure format")
    categorization: Optional[List[str]] = Field(default=None, description="Knowledge categories")
    relationships: Optional[Dict[str, List[str]]] = Field(default=None, description="Knowledge relationships")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    validation_rules: Optional[List[str]] = Field(default=None, description="Validation rules")


class CodifierAgent(EnhancedBaseAgent[DocumentationRequest, Dict[str, Any]]):
    """
    Specialized agent for documentation, logging, and knowledge codification.

    Capabilities:
    - Generate comprehensive documentation from code and specifications
    - Structure and analyze log data for insights
    - Codify knowledge into structured formats
    - Maintain documentation consistency and standards
    - Create process documentation and runbooks
    - Extract and organize information from various sources
    """

    def __init__(
        self,
        agent_id: str,
        name: str = "CodifierAgent",
        description: str = "Specialized agent for documentation and knowledge codification",
        **kwargs
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            agent_type=AgentType.CODIFIER_AGENT,
            **kwargs
        )

        # Documentation configuration
        self.supported_formats = {
            "markdown", "html", "pdf", "docx", "rst", "asciidoc", "latex"
        }

        self.documentation_templates = {
            "api": "API Reference Template",
            "user_guide": "User Guide Template",
            "technical_spec": "Technical Specification Template",
            "runbook": "Operational Runbook Template",
            "architecture": "Architecture Documentation Template"
        }

        # Logging configuration
        self.log_parsers = {
            "apache": r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\S+)',
            "nginx": r'(?P<ip>\S+) - \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\d+)',
            "syslog": r'(?P<timestamp>\w{3} \d{2} \d{2}:\d{2}:\d{2}) (?P<host>\S+) (?P<process>\S+): (?P<message>.*)',
            "json": r'^\{.*\}$',
            "application": r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?P<level>\w+)\] (?P<message>.*)'
        }

        # Knowledge codification patterns
        self.knowledge_schemas = {
            "process": {
                "name": str,
                "description": str,
                "steps": list,
                "prerequisites": list,
                "outcomes": list,
                "metadata": dict
            },
            "technical": {
                "component": str,
                "description": str,
                "interfaces": list,
                "dependencies": list,
                "configuration": dict,
                "documentation": str
            },
            "business": {
                "process_name": str,
                "stakeholders": list,
                "inputs": list,
                "outputs": list,
                "rules": list,
                "metrics": dict
            }
        }

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "documentation_generation",
            "code_documentation",
            "api_documentation",
            "log_analysis",
            "log_structuring",
            "knowledge_codification",
            "process_documentation",
            "runbook_creation",
            "content_standardization",
            "information_extraction",
            "taxonomy_creation",
            "metadata_generation",
            "documentation_validation",
            "content_formatting",
            "template_management"
        ]

    async def _execute_core_logic(
        self,
        request: Union[DocumentationRequest, LogStructuringRequest, KnowledgeCodeRequest],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the core documentation/codification logic."""

        with logfire.span("CodifierAgent execution", request_type=type(request).__name__):
            try:
                if isinstance(request, DocumentationRequest):
                    return await self._handle_documentation_request(request, context)
                elif isinstance(request, LogStructuringRequest):
                    return await self._handle_log_structuring_request(request, context)
                elif isinstance(request, KnowledgeCodeRequest):
                    return await self._handle_knowledge_codification_request(request, context)
                else:
                    # Handle generic documentation requests
                    return await self._handle_generic_request(request, context)

            except Exception as e:
                logfire.error("CodifierAgent execution failed", error=str(e))
                raise AgentExecutionError(f"Codification failed: {str(e)}")

    async def _handle_documentation_request(
        self,
        request: DocumentationRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle documentation generation requests."""

        with logfire.span("Documentation generation", doc_type=request.documentation_type.value):
            result = {
                "documentation_type": request.documentation_type.value,
                "output_format": request.output_format,
                "generated_at": datetime.utcnow().isoformat(),
                "metadata": {}
            }

            # Generate documentation based on type
            if request.documentation_type == DocumentationType.API_DOCUMENTATION:
                result.update(await self._generate_api_documentation(request, context))
            elif request.documentation_type == DocumentationType.CODE_DOCUMENTATION:
                result.update(await self._generate_code_documentation(request, context))
            elif request.documentation_type == DocumentationType.RUNBOOK:
                result.update(await self._generate_runbook(request, context))
            elif request.documentation_type == DocumentationType.TECHNICAL_SPECIFICATION:
                result.update(await self._generate_technical_spec(request, context))
            else:
                result.update(await self._generate_generic_documentation(request, context))

            logfire.info("Documentation generated",
                        doc_type=request.documentation_type.value,
                        output_size=len(result.get("content", "")))

            return result

    async def _handle_log_structuring_request(
        self,
        request: LogStructuringRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle log analysis and structuring requests."""

        with logfire.span("Log structuring", source_format=request.source_format.value):
            result = {
                "source_format": request.source_format.value,
                "target_format": request.target_format.value,
                "processed_at": datetime.utcnow().isoformat(),
                "statistics": {}
            }

            # Parse and structure log data
            parsed_logs = await self._parse_log_data(request.log_data, request.source_format)

            # Apply filters if specified
            if request.filter_criteria:
                parsed_logs = await self._filter_logs(parsed_logs, request.filter_criteria)

            # Extract specific fields if requested
            if request.extract_fields:
                parsed_logs = await self._extract_log_fields(parsed_logs, request.extract_fields)

            # Aggregate data if requested
            if request.aggregate_data:
                aggregated_data = await self._aggregate_log_data(parsed_logs)
                result["aggregated_data"] = aggregated_data

            # Format output
            result["structured_logs"] = await self._format_log_output(parsed_logs, request.target_format)
            result["statistics"] = {
                "total_entries": len(parsed_logs),
                "unique_sources": len(set(log.get("source", "unknown") for log in parsed_logs)),
                "date_range": self._get_log_date_range(parsed_logs)
            }

            logfire.info("Log structuring completed",
                        entries_processed=len(parsed_logs),
                        target_format=request.target_format.value)

            return result

    async def _handle_knowledge_codification_request(
        self,
        request: KnowledgeCodeRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle knowledge codification requests."""

        with logfire.span("Knowledge codification", knowledge_type=request.knowledge_type):
            result = {
                "knowledge_type": request.knowledge_type,
                "output_structure": request.output_structure,
                "codified_at": datetime.utcnow().isoformat(),
                "validation_results": {}
            }

            # Process knowledge sources
            processed_knowledge = []
            sources = request.knowledge_source if isinstance(request.knowledge_source, list) else [request.knowledge_source]

            for source in sources:
                codified = await self._codify_knowledge_source(source, request.knowledge_type, context)
                processed_knowledge.append(codified)

            # Apply categorization if specified
            if request.categorization:
                processed_knowledge = await self._categorize_knowledge(processed_knowledge, request.categorization)

            # Establish relationships if specified
            if request.relationships:
                processed_knowledge = await self._establish_knowledge_relationships(
                    processed_knowledge, request.relationships
                )

            # Validate against rules if specified
            if request.validation_rules:
                validation_results = await self._validate_knowledge(processed_knowledge, request.validation_rules)
                result["validation_results"] = validation_results

            # Structure output
            result["codified_knowledge"] = await self._structure_knowledge_output(
                processed_knowledge, request.output_structure
            )

            result["metadata"] = {
                "sources_processed": len(sources),
                "knowledge_items": len(processed_knowledge),
                "categories": request.categorization or [],
                **request.metadata or {}
            }

            logfire.info("Knowledge codification completed",
                        knowledge_type=request.knowledge_type,
                        items_processed=len(processed_knowledge))

            return result

    async def _generate_api_documentation(
        self,
        request: DocumentationRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate API documentation from source code or specifications."""

        # Placeholder implementation - would integrate with actual API discovery
        content = f"""# API Documentation

## Overview
This API provides {request.content or 'comprehensive functionality'}.

## Authentication
Details about authentication mechanisms.

## Endpoints
List of available endpoints with descriptions.

## Examples
{'Code examples included.' if request.include_examples else 'Contact support for examples.'}

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return {
            "content": content,
            "sections": ["overview", "authentication", "endpoints", "examples"],
            "word_count": len(content.split()),
            "estimated_read_time": len(content.split()) // 200  # minutes
        }

    async def _generate_code_documentation(
        self,
        request: DocumentationRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate code documentation from source files."""

        # Placeholder implementation - would analyze actual source code
        content = f"""# Code Documentation

## Module Overview
{request.content or 'Module documentation'}

## Classes and Functions
Detailed documentation of classes and functions.

## Usage Examples
{'Examples provided.' if request.include_examples else 'See source code for usage.'}

## Dependencies
List of module dependencies.

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return {
            "content": content,
            "sections": ["overview", "classes", "functions", "examples", "dependencies"],
            "coverage": 85.5,  # Percentage of code documented
            "missing_docs": []
        }

    async def _generate_runbook(
        self,
        request: DocumentationRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate operational runbook documentation."""

        content = f"""# Operational Runbook

## Purpose
{request.content or 'Operational procedures and troubleshooting guide'}

## Prerequisites
- System access requirements
- Required tools and permissions

## Procedures
Step-by-step operational procedures.

## Troubleshooting
Common issues and resolution steps.

## Contacts
Emergency contacts and escalation procedures.

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return {
            "content": content,
            "sections": ["purpose", "prerequisites", "procedures", "troubleshooting", "contacts"],
            "procedure_count": 5,
            "urgency_level": "operational"
        }

    async def _generate_technical_spec(
        self,
        request: DocumentationRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate technical specification documentation."""

        content = f"""# Technical Specification

## Scope
{request.content or 'Technical specification and requirements'}

## Architecture
{'System architecture diagrams.' if request.include_diagrams else 'Architecture overview.'}

## Requirements
Functional and non-functional requirements.

## Implementation Details
Technical implementation specifications.

## Testing Strategy
Testing approach and validation criteria.

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return {
            "content": content,
            "sections": ["scope", "architecture", "requirements", "implementation", "testing"],
            "requirement_count": 10,
            "complexity_level": "high"
        }

    async def _generate_generic_documentation(
        self,
        request: DocumentationRequest,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate generic documentation based on request parameters."""

        content = f"""# {request.documentation_type.value.replace('_', ' ').title()}

## Content
{request.content or 'Documentation content'}

## Details
Generated documentation for {request.target_audience} audience.

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return {
            "content": content,
            "sections": ["content", "details"],
            "target_audience": request.target_audience,
            "language": request.language
        }

    async def _parse_log_data(self, log_data: Union[str, List[str], Dict[str, Any]], source_format: LogFormat) -> List[Dict[str, Any]]:
        """Parse log data based on source format."""

        parsed_logs = []

        if source_format == LogFormat.JSON:
            if isinstance(log_data, str):
                try:
                    parsed_logs = [json.loads(log_data)]
                except json.JSONDecodeError:
                    # Handle multi-line JSON logs
                    for line in log_data.split('\n'):
                        if line.strip():
                            try:
                                parsed_logs.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            elif isinstance(log_data, list):
                for item in log_data:
                    if isinstance(item, str):
                        try:
                            parsed_logs.append(json.loads(item))
                        except json.JSONDecodeError:
                            continue
                    else:
                        parsed_logs.append(item)
            else:
                parsed_logs = [log_data]

        elif source_format == LogFormat.STRUCTURED:
            # Handle structured logs with known patterns
            if isinstance(log_data, str):
                for line in log_data.split('\n'):
                    if line.strip():
                        parsed_logs.append(self._parse_structured_log_line(line))
            elif isinstance(log_data, list):
                for line in log_data:
                    parsed_logs.append(self._parse_structured_log_line(str(line)))

        else:  # Plain text or other formats
            if isinstance(log_data, str):
                for line in log_data.split('\n'):
                    if line.strip():
                        parsed_logs.append({"raw": line.strip(), "timestamp": datetime.utcnow().isoformat()})
            elif isinstance(log_data, list):
                for line in log_data:
                    parsed_logs.append({"raw": str(line), "timestamp": datetime.utcnow().isoformat()})

        return parsed_logs

    def _parse_structured_log_line(self, line: str) -> Dict[str, Any]:
        """Parse a structured log line using known patterns."""

        for pattern_name, pattern in self.log_parsers.items():
            match = re.match(pattern, line)
            if match:
                result = match.groupdict()
                result["parser"] = pattern_name
                result["raw"] = line
                return result

        # Fallback for unrecognized patterns
        return {"raw": line, "timestamp": datetime.utcnow().isoformat(), "parser": "unknown"}

    async def _filter_logs(self, logs: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter logs based on specified criteria."""

        filtered_logs = []

        for log in logs:
            matches = True
            for key, value in criteria.items():
                if key not in log:
                    matches = False
                    break

                if isinstance(value, str) and value not in str(log[key]):
                    matches = False
                    break
                elif isinstance(value, (int, float)) and log[key] != value:
                    matches = False
                    break
                elif isinstance(value, list) and log[key] not in value:
                    matches = False
                    break

            if matches:
                filtered_logs.append(log)

        return filtered_logs

    async def _extract_log_fields(self, logs: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
        """Extract specific fields from logs."""

        extracted_logs = []

        for log in logs:
            extracted = {}
            for field in fields:
                if field in log:
                    extracted[field] = log[field]
                else:
                    extracted[field] = None
            extracted_logs.append(extracted)

        return extracted_logs

    async def _aggregate_log_data(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate log data for analysis."""

        aggregation = {
            "total_entries": len(logs),
            "field_counts": {},
            "value_distributions": {},
            "time_distribution": {}
        }

        # Count field occurrences
        for log in logs:
            for field in log.keys():
                aggregation["field_counts"][field] = aggregation["field_counts"].get(field, 0) + 1

        # Analyze value distributions for key fields
        common_fields = ["level", "status", "method", "host"]
        for field in common_fields:
            if field in aggregation["field_counts"]:
                distribution = {}
                for log in logs:
                    if field in log:
                        value = str(log[field])
                        distribution[value] = distribution.get(value, 0) + 1
                aggregation["value_distributions"][field] = distribution

        return aggregation

    async def _format_log_output(self, logs: List[Dict[str, Any]], target_format: LogFormat) -> Union[str, List[Dict[str, Any]]]:
        """Format log output in the target format."""

        if target_format == LogFormat.JSON:
            return logs
        elif target_format == LogFormat.CSV:
            if not logs:
                return ""

            # Get all unique fields
            all_fields = set()
            for log in logs:
                all_fields.update(log.keys())

            # Create CSV content
            csv_lines = [",".join(sorted(all_fields))]
            for log in logs:
                values = [str(log.get(field, "")) for field in sorted(all_fields)]
                csv_lines.append(",".join(values))

            return "\n".join(csv_lines)

        elif target_format == LogFormat.MARKDOWN:
            if not logs:
                return "No logs to display"

            # Create markdown table
            all_fields = set()
            for log in logs:
                all_fields.update(log.keys())

            fields = sorted(all_fields)
            lines = [
                "| " + " | ".join(fields) + " |",
                "| " + " | ".join(["---"] * len(fields)) + " |"
            ]

            for log in logs:
                values = [str(log.get(field, "")) for field in fields]
                lines.append("| " + " | ".join(values) + " |")

            return "\n".join(lines)

        else:  # Plain text
            lines = []
            for log in logs:
                lines.append(json.dumps(log, indent=2))
            return "\n".join(lines)

    def _get_log_date_range(self, logs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get the date range of logs."""

        timestamps = []
        for log in logs:
            if "timestamp" in log:
                timestamps.append(log["timestamp"])

        if timestamps:
            return {
                "earliest": min(timestamps),
                "latest": max(timestamps)
            }

        return {"earliest": None, "latest": None}

    async def _codify_knowledge_source(self, source: str, knowledge_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Codify a single knowledge source."""

        schema = self.knowledge_schemas.get(knowledge_type, {})

        codified = {
            "source": source,
            "knowledge_type": knowledge_type,
            "codified_at": datetime.utcnow().isoformat(),
            "schema_version": "1.0"
        }

        # Apply schema structure
        for field, field_type in schema.items():
            if field_type == str:
                codified[field] = f"Extracted {field} from {source}"
            elif field_type == list:
                codified[field] = [f"Item 1 for {field}", f"Item 2 for {field}"]
            elif field_type == dict:
                codified[field] = {f"{field}_key": f"{field}_value"}

        return codified

    async def _categorize_knowledge(self, knowledge_items: List[Dict[str, Any]], categories: List[str]) -> List[Dict[str, Any]]:
        """Categorize knowledge items."""

        for item in knowledge_items:
            # Simple categorization logic - in practice would use ML or rules
            item["categories"] = categories[:2]  # Assign first two categories
            item["primary_category"] = categories[0] if categories else "uncategorized"

        return knowledge_items

    async def _establish_knowledge_relationships(
        self,
        knowledge_items: List[Dict[str, Any]],
        relationships: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Establish relationships between knowledge items."""

        for item in knowledge_items:
            item["relationships"] = {}
            for rel_type, related_items in relationships.items():
                item["relationships"][rel_type] = related_items

        return knowledge_items

    async def _validate_knowledge(self, knowledge_items: List[Dict[str, Any]], validation_rules: List[str]) -> Dict[str, Any]:
        """Validate codified knowledge against rules."""

        results = {
            "total_items": len(knowledge_items),
            "validated_items": 0,
            "validation_errors": [],
            "warnings": []
        }

        for i, item in enumerate(knowledge_items):
            item_valid = True

            for rule in validation_rules:
                if "required_field" in rule:
                    field = rule.split(":")[-1].strip()
                    if field not in item:
                        results["validation_errors"].append(f"Item {i}: Missing required field '{field}'")
                        item_valid = False

                elif "min_length" in rule:
                    # Example validation rule
                    if len(str(item.get("description", ""))) < 10:
                        results["warnings"].append(f"Item {i}: Description too short")

            if item_valid:
                results["validated_items"] += 1

        return results

    async def _structure_knowledge_output(self, knowledge_items: List[Dict[str, Any]], output_structure: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Structure knowledge output according to specified format."""

        if output_structure == "hierarchical":
            structured = {
                "knowledge_base": {
                    "metadata": {
                        "total_items": len(knowledge_items),
                        "generated_at": datetime.utcnow().isoformat()
                    },
                    "categories": {},
                    "items": knowledge_items
                }
            }

            # Group by categories
            for item in knowledge_items:
                category = item.get("primary_category", "uncategorized")
                if category not in structured["knowledge_base"]["categories"]:
                    structured["knowledge_base"]["categories"][category] = []
                structured["knowledge_base"]["categories"][category].append(item.get("source", "unknown"))

            return structured

        elif output_structure == "flat":
            return knowledge_items

        else:  # structured (default)
            return {
                "knowledge_items": knowledge_items,
                "summary": {
                    "total_items": len(knowledge_items),
                    "knowledge_types": list(set(item.get("knowledge_type") for item in knowledge_items)),
                    "generated_at": datetime.utcnow().isoformat()
                }
            }

    async def _handle_generic_request(self, request: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic documentation/codification requests."""

        return {
            "result": "Generic codification completed",
            "request_type": type(request).__name__,
            "processed_at": datetime.utcnow().isoformat(),
            "context_keys": list(context.keys()) if context else []
        }

    def get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration for the CodifierAgent."""
        return {
            "max_document_size": 10000,  # Maximum document size in lines
            "supported_formats": list(self.supported_formats),
            "default_template": "standard",
            "auto_generate_toc": True,
            "include_metadata": True,
            "log_retention_days": 30,
            "knowledge_validation": True,
            "output_compression": False
        }

    async def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate agent configuration."""
        required_keys = ["max_document_size", "supported_formats"]

        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required configuration key: {key}")

        if config["max_document_size"] <= 0:
            raise ValidationError("max_document_size must be positive")

        return True
