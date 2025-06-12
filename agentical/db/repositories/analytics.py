"""
Analytics Repository for Agentical

This module provides comprehensive data access for analytics and monitoring,
including metrics collection, performance analysis, and observability data.

Features:
- Workflow execution analytics
- Agent performance metrics
- System resource monitoring
- Logfire integration for observability
- Custom analytics queries
- Data export capabilities
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from sqlalchemy import text, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import logfire

from ..models.workflow import (
    Workflow, WorkflowExecution, WorkflowStepExecution,
    ExecutionStatus, StepStatus
)
from ..models.agent import Agent, AgentExecution
from ..models.base import BaseModel
from ...core.exceptions import AnalyticsError, DatabaseError


class AsyncAnalyticsRepository:
    """Repository for analytics and monitoring data."""

    def __init__(self, db_session: AsyncSession):
        """Initialize repository with database session."""
        self.db = db_session

    async def health_check(self) -> bool:
        """Perform database health check."""
        try:
            result = await self.db.execute(text("SELECT 1"))
            return result.scalar() == 1
        except Exception as e:
            logfire.error("Analytics repository health check failed", error=str(e))
            raise DatabaseError(f"Database health check failed: {str(e)}")

    async def get_workflow_metrics(
        self,
        from_date: datetime,
        to_date: datetime,
        workflow_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive workflow execution metrics."""
        try:
            # Base query for workflow executions
            base_query = """
            SELECT
                we.status,
                we.started_at,
                we.completed_at,
                we.workflow_id,
                w.name as workflow_name,
                EXTRACT(EPOCH FROM (we.completed_at - we.started_at)) as duration_seconds
            FROM workflow_executions we
            JOIN workflows w ON w.id = we.workflow_id
            WHERE we.created_at BETWEEN :from_date AND :to_date
            """

            params = {"from_date": from_date, "to_date": to_date}

            if workflow_ids:
                base_query += " AND we.workflow_id = ANY(:workflow_ids)"
                params["workflow_ids"] = workflow_ids

            result = await self.db.execute(text(base_query), params)
            executions = result.fetchall()

            # Calculate metrics
            total_executions = len(executions)
            successful_executions = len([e for e in executions if e.status == ExecutionStatus.COMPLETED.value])
            failed_executions = len([e for e in executions if e.status == ExecutionStatus.FAILED.value])

            # Calculate average duration (only for completed executions)
            completed_with_duration = [
                e for e in executions
                if e.status == ExecutionStatus.COMPLETED.value and e.duration_seconds is not None
            ]
            average_duration_seconds = (
                sum(e.duration_seconds for e in completed_with_duration) / len(completed_with_duration)
                if completed_with_duration else 0.0
            )

            success_rate_percent = (successful_executions / total_executions * 100) if total_executions > 0 else 0.0

            # Get executions per hour
            executions_per_hour = await self._get_executions_per_hour(from_date, to_date, workflow_ids)

            # Get most used workflows
            workflow_usage = defaultdict(int)
            workflow_names = {}
            for execution in executions:
                workflow_usage[execution.workflow_id] += 1
                workflow_names[execution.workflow_id] = execution.workflow_name

            most_used_workflows = [
                {
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_names[workflow_id],
                    "execution_count": count
                }
                for workflow_id, count in sorted(workflow_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            ]

            # Get error distribution
            error_distribution = await self._get_workflow_error_distribution(from_date, to_date, workflow_ids)

            # Get performance trends
            performance_trends = await self._get_workflow_performance_trends(from_date, to_date, workflow_ids)

            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "average_duration_seconds": average_duration_seconds,
                "success_rate_percent": success_rate_percent,
                "executions_per_hour": executions_per_hour,
                "most_used_workflows": most_used_workflows,
                "error_distribution": error_distribution,
                "performance_trends": performance_trends
            }

        except Exception as e:
            logfire.error("Failed to get workflow metrics", error=str(e))
            raise AnalyticsError(f"Failed to get workflow metrics: {str(e)}")

    async def get_agent_metrics(
        self,
        from_date: datetime,
        to_date: datetime,
        agent_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive agent execution metrics."""
        try:
            # Base query for agent executions
            base_query = """
            SELECT
                ae.status,
                ae.started_at,
                ae.completed_at,
                ae.agent_id,
                a.name as agent_name,
                a.agent_type,
                EXTRACT(EPOCH FROM (ae.completed_at - ae.started_at)) * 1000 as response_time_ms
            FROM agent_executions ae
            JOIN agents a ON a.id = ae.agent_id
            WHERE ae.created_at BETWEEN :from_date AND :to_date
            """

            params = {"from_date": from_date, "to_date": to_date}

            if agent_types:
                base_query += " AND a.agent_type = ANY(:agent_types)"
                params["agent_types"] = agent_types

            result = await self.db.execute(text(base_query), params)
            executions = result.fetchall()

            # Calculate metrics
            total_executions = len(executions)
            successful_executions = len([e for e in executions if e.status == "completed"])
            failed_executions = len([e for e in executions if e.status == "failed"])

            # Calculate average response time (only for completed executions)
            completed_with_time = [
                e for e in executions
                if e.status == "completed" and e.response_time_ms is not None
            ]
            average_response_time_ms = (
                sum(e.response_time_ms for e in completed_with_time) / len(completed_with_time)
                if completed_with_time else 0.0
            )

            success_rate_percent = (successful_executions / total_executions * 100) if total_executions > 0 else 0.0

            # Get most active agents
            agent_usage = defaultdict(int)
            agent_info = {}
            for execution in executions:
                agent_usage[execution.agent_id] += 1
                agent_info[execution.agent_id] = {
                    "name": execution.agent_name,
                    "type": execution.agent_type
                }

            most_active_agents = [
                {
                    "agent_id": agent_id,
                    "agent_name": agent_info[agent_id]["name"],
                    "agent_type": agent_info[agent_id]["type"],
                    "execution_count": count
                }
                for agent_id, count in sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            ]

            # Get error types
            error_types = await self._get_agent_error_types(from_date, to_date, agent_types)

            # Get performance by agent type
            performance_by_agent_type = await self._get_agent_performance_by_type(from_date, to_date, agent_types)

            return {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "average_response_time_ms": average_response_time_ms,
                "success_rate_percent": success_rate_percent,
                "most_active_agents": most_active_agents,
                "error_types": error_types,
                "performance_by_agent_type": performance_by_agent_type
            }

        except Exception as e:
            logfire.error("Failed to get agent metrics", error=str(e))
            raise AnalyticsError(f"Failed to get agent metrics: {str(e)}")

    async def get_logfire_metrics(
        self,
        from_date: datetime,
        to_date: datetime,
        service_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get Logfire observability metrics."""
        try:
            # This would integrate with Logfire API to get actual metrics
            # For now, return mock data structure

            # In a real implementation, you would:
            # 1. Connect to Logfire API
            # 2. Query spans and traces for the time range
            # 3. Calculate metrics from the data

            return {
                "total_spans": 15420,
                "error_spans": 234,
                "average_span_duration_ms": 125.5,
                "spans_by_service": {
                    "agentical-api": 8500,
                    "workflow-engine": 4200,
                    "agent-executor": 2720
                },
                "error_rate_percent": 1.52,
                "top_errors": [
                    {
                        "error_type": "ValidationError",
                        "count": 89,
                        "percentage": 38.0
                    },
                    {
                        "error_type": "TimeoutError",
                        "count": 67,
                        "percentage": 28.6
                    },
                    {
                        "error_type": "ConnectionError",
                        "count": 45,
                        "percentage": 19.2
                    }
                ],
                "performance_insights": [
                    {
                        "insight": "Workflow execution time increased 15% over past week",
                        "severity": "medium",
                        "recommendation": "Review workflow optimization"
                    },
                    {
                        "insight": "Agent response time variance high for NLP agents",
                        "severity": "low",
                        "recommendation": "Consider load balancing"
                    }
                ],
                "trace_analysis": {
                    "average_trace_duration_ms": 892.3,
                    "slowest_operations": [
                        {"operation": "database_query", "avg_duration_ms": 245.6},
                        {"operation": "llm_inference", "avg_duration_ms": 1850.2},
                        {"operation": "file_processing", "avg_duration_ms": 156.8}
                    ]
                }
            }

        except Exception as e:
            logfire.error("Failed to get Logfire metrics", error=str(e))
            raise AnalyticsError(f"Failed to get Logfire metrics: {str(e)}")

    async def get_available_metrics(self) -> List[str]:
        """Get list of available metrics for custom queries."""
        return [
            "workflow_executions",
            "workflow_duration",
            "workflow_success_rate",
            "agent_executions",
            "agent_response_time",
            "agent_success_rate",
            "system_cpu_usage",
            "system_memory_usage",
            "system_disk_usage",
            "logfire_spans",
            "logfire_errors",
            "logfire_performance"
        ]

    async def query_metrics(
        self,
        metrics: List[str],
        time_range: Tuple[datetime, datetime],
        granularity: str = "1h",
        filters: Optional[Dict[str, Any]] = None,
        group_by: Optional[List[str]] = None,
        aggregation: str = "avg"
    ) -> List[Dict[str, Any]]:
        """Execute custom metrics query."""
        try:
            from_date, to_date = time_range
            filters = filters or {}
            group_by = group_by or []

            # Generate time buckets based on granularity
            time_buckets = self._generate_time_buckets(from_date, to_date, granularity)

            results = []

            for metric in metrics:
                metric_data = await self._query_single_metric(
                    metric, from_date, to_date, granularity,
                    filters, group_by, aggregation
                )
                results.extend(metric_data)

            return results

        except Exception as e:
            logfire.error("Failed to execute custom metrics query", error=str(e))
            raise AnalyticsError(f"Failed to execute metrics query: {str(e)}")

    async def export_metrics(
        self,
        from_date: datetime,
        to_date: datetime,
        metric_types: List[str],
        format: str = "json"
    ) -> str:
        """Export metrics data in specified format."""
        try:
            export_data = {}

            # Collect requested metric types
            if "system" in metric_types:
                export_data["system_metrics"] = await self._get_system_metrics_export(from_date, to_date)

            if "workflows" in metric_types:
                export_data["workflow_metrics"] = await self.get_workflow_metrics(from_date, to_date)

            if "agents" in metric_types:
                export_data["agent_metrics"] = await self.get_agent_metrics(from_date, to_date)

            if "logfire" in metric_types:
                export_data["logfire_metrics"] = await self.get_logfire_metrics(from_date, to_date)

            # Format data based on requested format
            if format == "json":
                return json.dumps(export_data, indent=2, default=str)
            elif format == "csv":
                return self._convert_to_csv(export_data)
            elif format == "prometheus":
                return self._convert_to_prometheus(export_data)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logfire.error("Failed to export metrics", error=str(e))
            raise AnalyticsError(f"Failed to export metrics: {str(e)}")

    # Private helper methods
    async def _get_executions_per_hour(
        self,
        from_date: datetime,
        to_date: datetime,
        workflow_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get workflow executions per hour."""
        query = """
        SELECT
            DATE_TRUNC('hour', created_at) as hour,
            COUNT(*) as execution_count
        FROM workflow_executions
        WHERE created_at BETWEEN :from_date AND :to_date
        """

        params = {"from_date": from_date, "to_date": to_date}

        if workflow_ids:
            query += " AND workflow_id = ANY(:workflow_ids)"
            params["workflow_ids"] = workflow_ids

        query += " GROUP BY DATE_TRUNC('hour', created_at) ORDER BY hour"

        result = await self.db.execute(text(query), params)
        return [{"hour": row.hour.isoformat(), "count": row.execution_count} for row in result.fetchall()]

    async def _get_workflow_error_distribution(
        self,
        from_date: datetime,
        to_date: datetime,
        workflow_ids: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Get distribution of workflow errors."""
        query = """
        SELECT
            COALESCE(error_message, 'Unknown Error') as error_type,
            COUNT(*) as error_count
        FROM workflow_executions
        WHERE created_at BETWEEN :from_date AND :to_date
        AND status = 'failed'
        """

        params = {"from_date": from_date, "to_date": to_date}

        if workflow_ids:
            query += " AND workflow_id = ANY(:workflow_ids)"
            params["workflow_ids"] = workflow_ids

        query += " GROUP BY error_message ORDER BY error_count DESC LIMIT 10"

        result = await self.db.execute(text(query), params)
        return {row.error_type: row.error_count for row in result.fetchall()}

    async def _get_workflow_performance_trends(
        self,
        from_date: datetime,
        to_date: datetime,
        workflow_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get workflow performance trends over time."""
        query = """
        SELECT
            DATE_TRUNC('day', created_at) as day,
            AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds,
            COUNT(*) as execution_count,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::float / COUNT(*) * 100 as success_rate
        FROM workflow_executions
        WHERE created_at BETWEEN :from_date AND :to_date
        AND started_at IS NOT NULL AND completed_at IS NOT NULL
        """

        params = {"from_date": from_date, "to_date": to_date}

        if workflow_ids:
            query += " AND workflow_id = ANY(:workflow_ids)"
            params["workflow_ids"] = workflow_ids

        query += " GROUP BY DATE_TRUNC('day', created_at) ORDER BY day"

        result = await self.db.execute(text(query), params)
        return [
            {
                "date": row.day.isoformat(),
                "avg_duration_seconds": float(row.avg_duration_seconds or 0),
                "execution_count": row.execution_count,
                "success_rate": float(row.success_rate or 0)
            }
            for row in result.fetchall()
        ]

    async def _get_agent_error_types(
        self,
        from_date: datetime,
        to_date: datetime,
        agent_types: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Get distribution of agent error types."""
        query = """
        SELECT
            COALESCE(ae.error_message, 'Unknown Error') as error_type,
            COUNT(*) as error_count
        FROM agent_executions ae
        JOIN agents a ON a.id = ae.agent_id
        WHERE ae.created_at BETWEEN :from_date AND :to_date
        AND ae.status = 'failed'
        """

        params = {"from_date": from_date, "to_date": to_date}

        if agent_types:
            query += " AND a.agent_type = ANY(:agent_types)"
            params["agent_types"] = agent_types

        query += " GROUP BY ae.error_message ORDER BY error_count DESC LIMIT 10"

        result = await self.db.execute(text(query), params)
        return {row.error_type: row.error_count for row in result.fetchall()}

    async def _get_agent_performance_by_type(
        self,
        from_date: datetime,
        to_date: datetime,
        agent_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get agent performance metrics by agent type."""
        query = """
        SELECT
            a.agent_type,
            COUNT(*) as total_executions,
            SUM(CASE WHEN ae.status = 'completed' THEN 1 ELSE 0 END) as successful_executions,
            AVG(EXTRACT(EPOCH FROM (ae.completed_at - ae.started_at)) * 1000) as avg_response_time_ms
        FROM agent_executions ae
        JOIN agents a ON a.id = ae.agent_id
        WHERE ae.created_at BETWEEN :from_date AND :to_date
        AND ae.started_at IS NOT NULL AND ae.completed_at IS NOT NULL
        """

        params = {"from_date": from_date, "to_date": to_date}

        if agent_types:
            query += " AND a.agent_type = ANY(:agent_types)"
            params["agent_types"] = agent_types

        query += " GROUP BY a.agent_type ORDER BY total_executions DESC"

        result = await self.db.execute(text(query), params)
        return [
            {
                "agent_type": row.agent_type,
                "total_executions": row.total_executions,
                "successful_executions": row.successful_executions,
                "success_rate": (row.successful_executions / row.total_executions * 100) if row.total_executions > 0 else 0,
                "avg_response_time_ms": float(row.avg_response_time_ms or 0)
            }
            for row in result.fetchall()
        ]

    def _generate_time_buckets(self, from_date: datetime, to_date: datetime, granularity: str) -> List[datetime]:
        """Generate time buckets for the given granularity."""
        granularity_mapping = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "1d": timedelta(days=1)
        }

        delta = granularity_mapping.get(granularity, timedelta(hours=1))
        buckets = []
        current = from_date

        while current < to_date:
            buckets.append(current)
            current += delta

        return buckets

    async def _query_single_metric(
        self,
        metric: str,
        from_date: datetime,
        to_date: datetime,
        granularity: str,
        filters: Dict[str, Any],
        group_by: List[str],
        aggregation: str
    ) -> List[Dict[str, Any]]:
        """Query a single metric with specified parameters."""
        # This would implement metric-specific queries
        # For now, return mock data
        return [
            {
                "metric": metric,
                "timestamp": from_date.isoformat(),
                "value": 100.0,
                "aggregation": aggregation
            }
        ]

    async def _get_system_metrics_export(self, from_date: datetime, to_date: datetime) -> Dict[str, Any]:
        """Get system metrics for export."""
        # This would collect historical system metrics
        # For now, return current metrics structure
        return {
            "cpu_usage_percent": 45.2,
            "memory_usage_percent": 67.8,
            "disk_usage_percent": 34.1,
            "network_io_bytes": {"sent": 1024000, "received": 2048000}
        }

    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert metrics data to CSV format."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write headers
        writer.writerow(["metric_type", "metric_name", "value", "timestamp"])

        # Flatten nested data structure
        for metric_type, metrics in data.items():
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    writer.writerow([metric_type, key, value, datetime.utcnow().isoformat()])

        return output.getvalue()

    def _convert_to_prometheus(self, data: Dict[str, Any]) -> str:
        """Convert metrics data to Prometheus format."""
        lines = []
        timestamp = int(datetime.utcnow().timestamp() * 1000)

        for metric_type, metrics in data.items():
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        metric_name = f"agentical_{metric_type}_{key}"
                        lines.append(f"{metric_name} {value} {timestamp}")

        return "\n".join(lines)
