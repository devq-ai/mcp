"""
Database Profiler for Performance Monitoring

This module provides tools for profiling and monitoring database performance,
helping identify bottlenecks and optimize queries. It includes:

- Query profiling to track execution time and resource usage
- Performance analysis for slow queries
- Automated index recommendations
- Integration with Logfire for observability
- Performance metrics collection and reporting
"""

import time
import logging
import functools
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
import json
from collections import defaultdict

import logfire
from sqlalchemy import event, inspect, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Select
from sqlalchemy.exc import SQLAlchemyError

# Type variables
T = TypeVar('T')

# Configure logging
logger = logging.getLogger(__name__)

# Profiling configuration from environment variables
PROFILING_ENABLED = os.getenv("DB_PROFILING_ENABLED", "true").lower() == "true"
SLOW_QUERY_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD", "0.5"))  # seconds
VERY_SLOW_QUERY_THRESHOLD = float(os.getenv("VERY_SLOW_QUERY_THRESHOLD", "2.0"))  # seconds
QUERY_SAMPLE_RATE = float(os.getenv("QUERY_SAMPLE_RATE", "0.1"))  # Sample 10% of all queries
MAX_QUERY_HISTORY = int(os.getenv("MAX_QUERY_HISTORY", "100"))
PROFILE_LOG_PATH = os.getenv("PROFILE_LOG_PATH", "logs/database_profile.json")

# Store query statistics
query_stats = defaultdict(lambda: {
    "count": 0,
    "total_time": 0.0,
    "min_time": float("inf"),
    "max_time": 0.0,
    "avg_time": 0.0,
    "last_seen": None,
    "tables": set(),
    "parameters_sample": [],
})
query_stats_lock = threading.Lock()

# Store query history for slow queries
slow_query_history: List[Dict[str, Any]] = []
slow_query_history_lock = threading.Lock()

# Table access statistics
table_stats = defaultdict(lambda: {
    "reads": 0,
    "writes": 0,
    "deletes": 0,
    "total_time": 0.0,
    "slow_queries": 0,
})
table_stats_lock = threading.Lock()

# Index recommendation storage
index_recommendations: Dict[str, Dict[str, Any]] = {}
index_recommendations_lock = threading.Lock()


@dataclass
class QueryProfile:
    """Query profiling information."""
    
    query: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    execution_time: float = 0.0
    row_count: int = 0
    query_type: str = "unknown"
    tables: Set[str] = field(default_factory=set)
    database: str = "main"
    error: Optional[str] = None
    
    def complete(self, row_count: int = 0) -> None:
        """Complete the profile with execution statistics."""
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        self.row_count = row_count
    
    def is_slow(self) -> bool:
        """Check if this is a slow query."""
        return self.execution_time > SLOW_QUERY_THRESHOLD
    
    def is_very_slow(self) -> bool:
        """Check if this is a very slow query."""
        return self.execution_time > VERY_SLOW_QUERY_THRESHOLD
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "query": self.query,
            "parameters": {k: str(v) for k, v in self.parameters.items()},
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "execution_time": self.execution_time,
            "row_count": self.row_count,
            "query_type": self.query_type,
            "tables": list(self.tables),
            "database": self.database,
            "error": self.error
        }
    
    def log_if_slow(self) -> None:
        """Log query if it's slow."""
        if not self.is_slow():
            return
            
        profile_dict = self.to_dict()
        
        # Log slow query
        if self.is_very_slow():
            logfire.warning(
                "Very slow database query detected",
                execution_time=self.execution_time,
                query=self.query[:1000],  # Truncate very long queries
                query_type=self.query_type,
                tables=list(self.tables),
                row_count=self.row_count,
                threshold=VERY_SLOW_QUERY_THRESHOLD
            )
            logger.warning(
                f"Very slow query ({self.execution_time:.4f}s): {self.query[:1000]}"
            )
        else:
            logfire.info(
                "Slow database query detected",
                execution_time=self.execution_time,
                query=self.query[:1000],  # Truncate very long queries
                query_type=self.query_type,
                tables=list(self.tables),
                row_count=self.row_count,
                threshold=SLOW_QUERY_THRESHOLD
            )
            logger.info(
                f"Slow query ({self.execution_time:.4f}s): {self.query[:1000]}"
            )
        
        # Add to history
        with slow_query_history_lock:
            slow_query_history.append(profile_dict)
            
            # Maintain maximum history size
            if len(slow_query_history) > MAX_QUERY_HISTORY:
                slow_query_history.pop(0)
                
        # Update table statistics
        with table_stats_lock:
            for table in self.tables:
                table_stats[table]["slow_queries"] += 1
    
    def update_stats(self) -> None:
        """Update query statistics."""
        # Normalize query by removing literals
        norm_query = self.query.strip()
        
        with query_stats_lock:
            stats = query_stats[norm_query]
            stats["count"] += 1
            stats["total_time"] += self.execution_time
            stats["min_time"] = min(stats["min_time"], self.execution_time)
            stats["max_time"] = max(stats["max_time"], self.execution_time)
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["last_seen"] = datetime.now().isoformat()
            stats["tables"] = stats["tables"].union(self.tables)
            
            # Sample parameters for debugging
            if len(stats["parameters_sample"]) < 5:
                stats["parameters_sample"].append(self.parameters)
            
            # Update table access statistics
            with table_stats_lock:
                for table in self.tables:
                    if self.query_type == "SELECT":
                        table_stats[table]["reads"] += 1
                    elif self.query_type in ["INSERT", "UPDATE"]:
                        table_stats[table]["writes"] += 1
                    elif self.query_type == "DELETE":
                        table_stats[table]["deletes"] += 1
                    
                    table_stats[table]["total_time"] += self.execution_time


def extract_tables_from_query(query: str) -> Set[str]:
    """Extract table names from SQL query.
    
    This is a simple heuristic and may not work for all queries.
    A more robust solution would parse the SQL.
    
    Args:
        query: SQL query string
        
    Returns:
        Set of table names
    """
    query = query.lower()
    tables = set()
    
    # Look for FROM and JOIN clauses
    words = query.split()
    for i, word in enumerate(words):
        if word == "from" or word == "join" and i < len(words) - 1:
            # Get the table name, stripping any quotes or aliases
            table = words[i + 1].strip('";`')
            # Remove schema prefix if present
            if '.' in table:
                table = table.split('.')[-1]
            # Remove alias if present
            if ' ' in table:
                table = table.split(' ')[0]
            tables.add(table)
    
    return tables


def get_query_type(query: str) -> str:
    """Determine the type of SQL query.
    
    Args:
        query: SQL query string
        
    Returns:
        Query type (SELECT, INSERT, UPDATE, DELETE, etc.)
    """
    query = query.strip().upper()
    
    for stmt_type in ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]:
        if query.startswith(stmt_type):
            return stmt_type
    
    return "UNKNOWN"


@contextmanager
def profile_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryProfile:
    """Context manager for profiling database queries.
    
    Args:
        query: SQL query string
        parameters: Query parameters
        
    Yields:
        QueryProfile object
    """
    if not PROFILING_ENABLED:
        # Empty context manager if profiling disabled
        profile = QueryProfile(query=query)
        yield profile
        return
    
    # Initialize profile
    profile = QueryProfile(
        query=query,
        parameters=parameters or {},
        query_type=get_query_type(query),
        tables=extract_tables_from_query(query)
    )
    
    try:
        # Start timing
        yield profile
        
        # Complete profile
        profile.complete()
        
    except Exception as e:
        # Record error
        profile.complete()
        profile.error = str(e)
        raise
    finally:
        # Log if slow
        profile.log_if_slow()
        
        # Update statistics
        profile.update_stats()


def profile_sqlalchemy_query(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for profiling SQLAlchemy queries.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if not PROFILING_ENABLED:
            return func(*args, **kwargs)
        
        # Extract query from first SQLAlchemy object argument
        query_obj = None
        for arg in args:
            if hasattr(arg, 'statement'):
                query_obj = arg
                break
        
        if not query_obj:
            # No query object found, just call the function
            return func(*args, **kwargs)
        
        # Get query string and parameters
        compiled = query_obj.statement.compile(dialect=query_obj.session.bind.dialect)
        query_str = str(compiled)
        params = compiled.params
        
        with profile_query(query_str, params) as profile:
            result = func(*args, **kwargs)
            
            # Try to get row count from result
            try:
                if hasattr(result, "__len__"):
                    profile.row_count = len(result)
            except Exception:
                pass
            
            return result
    
    return wrapper


class QueryPerformanceAnalyzer:
    """Analyze query performance and recommend optimizations."""
    
    def __init__(self, session: Session):
        """Initialize with database session.
        
        Args:
            session: SQLAlchemy session
        """
        self.session = session
        self.inspector = inspect(session.bind)
        self.analyzed_tables: Set[str] = set()
        self.index_recommendations: Dict[str, Dict[str, Any]] = {}
    
    def analyze_table_indexes(self, table_name: str) -> Dict[str, Any]:
        """Analyze indexes on a table and recommend improvements.
        
        Args:
            table_name: Name of table to analyze
            
        Returns:
            Dictionary of recommendations
        """
        if table_name in self.analyzed_tables:
            return self.index_recommendations.get(table_name, {})
        
        # Mark as analyzed to avoid redundant work
        self.analyzed_tables.add(table_name)
        
        try:
            # Get existing indexes
            existing_indexes = self.inspector.get_indexes(table_name)
            existing_index_columns = set()
            for idx in existing_indexes:
                for col in idx['column_names']:
                    existing_index_columns.add(col)
            
            # Get table columns
            columns = self.inspector.get_columns(table_name)
            column_names = [col['name'] for col in columns]
            
            # Get table statistics from our tracking
            with table_stats_lock:
                table_stat = table_stats.get(table_name, {})
            
            recommendations = []
            
            # Check for missing indexes on primary key
            primary_keys = self.inspector.get_pk_constraint(table_name)['constrained_columns']
            if primary_keys and not any(idx['column_names'] == primary_keys for idx in existing_indexes):
                recommendations.append({
                    "type": "primary_key",
                    "columns": primary_keys,
                    "reason": "No index on primary key"
                })
            
            # Check for missing indexes on foreign keys
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            for fk in foreign_keys:
                fk_columns = fk['constrained_columns']
                if not any(set(idx['column_names']) == set(fk_columns) for idx in existing_indexes):
                    recommendations.append({
                        "type": "foreign_key",
                        "columns": fk_columns,
                        "references": fk['referred_table'],
                        "reason": f"No index on foreign key to {fk['referred_table']}"
                    })
            
            # Look for potential missing indexes based on query patterns
            # This would require more sophisticated analysis of query patterns
            # For now, we'll just make a simple recommendation based on access patterns
            
            result = {
                "table": table_name,
                "existing_indexes": existing_indexes,
                "recommendations": recommendations,
                "analysis_time": datetime.now().isoformat()
            }
            
            # Store recommendations
            self.index_recommendations[table_name] = result
            
            # Update global recommendations
            with index_recommendations_lock:
                index_recommendations[table_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing indexes for table {table_name}: {e}")
            return {
                "table": table_name,
                "error": str(e),
                "analysis_time": datetime.now().isoformat()
            }
    
    def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Analyze slow queries and suggest optimizations.
        
        Returns:
            List of optimization suggestions
        """
        with slow_query_history_lock:
            queries = slow_query_history.copy()
        
        optimizations = []
        
        for query_data in queries:
            query_type = query_data.get("query_type", "")
            tables = query_data.get("tables", [])
            
            if not tables:
                continue
            
            # Analyze each table involved in slow queries
            for table in tables:
                analysis = self.analyze_table_indexes(table)
                
                if analysis.get("recommendations"):
                    optimizations.append({
                        "query": query_data.get("query", "")[:100] + "...",
                        "execution_time": query_data.get("execution_time"),
                        "table": table,
                        "recommendations": analysis.get("recommendations", []),
                        "reason": "Table involved in slow query"
                    })
        
        return optimizations
    
    def get_table_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tables.
        
        Returns:
            Dictionary of table statistics
        """
        with table_stats_lock:
            stats = dict(table_stats)
        
        result = {}
        for table_name, table_stat in stats.items():
            # Get additional information about the table
            try:
                columns = self.inspector.get_columns(table_name)
                indexes = self.inspector.get_indexes(table_name)
                primary_key = self.inspector.get_pk_constraint(table_name)
                foreign_keys = self.inspector.get_foreign_keys(table_name)
                
                result[table_name] = {
                    **table_stat,
                    "column_count": len(columns),
                    "index_count": len(indexes),
                    "has_primary_key": bool(primary_key['constrained_columns']),
                    "foreign_key_count": len(foreign_keys)
                }
            except Exception as e:
                logger.error(f"Error getting table metadata for {table_name}: {e}")
                result[table_name] = table_stat
        
        return result
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get comprehensive optimization recommendations.
        
        Returns:
            Dictionary of recommendations
        """
        # Analyze all tables that have been accessed
        with table_stats_lock:
            tables = list(table_stats.keys())
        
        for table in tables:
            self.analyze_table_indexes(table)
        
        # Analyze slow queries
        query_optimizations = self.analyze_slow_queries()
        
        # Get overall statistics
        table_statistics = self.get_table_statistics()
        
        return {
            "tables": table_statistics,
            "slow_queries": len(slow_query_history),
            "query_optimizations": query_optimizations,
            "index_recommendations": self.index_recommendations,
            "generated_at": datetime.now().isoformat()
        }


def save_profile_log(file_path: Optional[str] = None) -> None:
    """Save profiling data to a log file.
    
    Args:
        file_path: Path to log file (defaults to PROFILE_LOG_PATH)
    """
    if not PROFILING_ENABLED:
        return
    
    file_path = file_path or PROFILE_LOG_PATH
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Collect data
    with query_stats_lock, table_stats_lock, slow_query_history_lock, index_recommendations_lock:
        data = {
            "timestamp": datetime.now().isoformat(),
            "query_stats": {
                k: {
                    **v,
                    "tables": list(v["tables"])
                } for k, v in dict(query_stats).items()
            },
            "table_stats": dict(table_stats),
            "slow_query_history": slow_query_history.copy(),
            "index_recommendations": dict(index_recommendations)
        }
    
    # Write to file
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Database profile saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving database profile to {file_path}: {e}")


def get_performance_metrics() -> Dict[str, Any]:
    """Get database performance metrics.
    
    Returns:
        Dictionary of performance metrics
    """
    if not PROFILING_ENABLED:
        return {"profiling_enabled": False}
    
    with query_stats_lock, table_stats_lock, slow_query_history_lock:
        # Calculate overall statistics
        total_queries = sum(stats["count"] for stats in query_stats.values())
        total_query_time = sum(stats["total_time"] for stats in query_stats.values())
        total_slow_queries = len(slow_query_history)
        
        # Calculate average query time
        avg_query_time = total_query_time / total_queries if total_queries > 0 else 0
        
        # Calculate slow query percentage
        slow_query_percentage = (total_slow_queries / total_queries) * 100 if total_queries > 0 else 0
        
        # Get top 5 slowest queries
        top_slow_queries = sorted(
            slow_query_history,
            key=lambda q: q.get("execution_time", 0),
            reverse=True
        )[:5]
        
        # Get top 5 most accessed tables
        top_tables = sorted(
            [(table, stats["reads"] + stats["writes"] + stats["deletes"])
             for table, stats in table_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "profiling_enabled": True,
            "total_queries": total_queries,
            "total_query_time": total_query_time,
            "avg_query_time": avg_query_time,
            "total_slow_queries": total_slow_queries,
            "slow_query_percentage": slow_query_percentage,
            "top_slow_queries": [
                {
                    "query": q.get("query", "")[:100] + "...",  # Truncate for readability
                    "execution_time": q.get("execution_time"),
                    "tables": q.get("tables"),
                    "query_type": q.get("query_type")
                }
                for q in top_slow_queries
            ],
            "top_tables": [
                {
                    "table": table,
                    "access_count": count,
                    "reads": table_stats[table]["reads"],
                    "writes": table_stats[table]["writes"],
                    "deletes": table_stats[table]["deletes"],
                    "slow_queries": table_stats[table]["slow_queries"]
                }
                for table, count in top_tables
            ],
            "timestamp": datetime.now().isoformat()
        }


# Export public API
__all__ = [
    "QueryProfile",
    "profile_query",
    "profile_sqlalchemy_query",
    "QueryPerformanceAnalyzer",
    "save_profile_log",
    "get_performance_metrics",
    "PROFILING_ENABLED",
    "SLOW_QUERY_THRESHOLD",
    "VERY_SLOW_QUERY_THRESHOLD"
]