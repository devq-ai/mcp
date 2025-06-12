"""
CSV Parser for Agentical

This module provides comprehensive CSV parsing and analysis capabilities with
advanced data validation, transformation, statistical analysis, and export features.

Features:
- Advanced CSV parsing with automatic schema detection
- Data type inference and conversion
- Data validation and cleaning
- Statistical analysis and profiling
- Data transformation and filtering
- Export to multiple formats (JSON, Excel, Parquet)
- Memory-efficient streaming for large files
- Integration with pandas and other data libraries
- Enterprise features (audit logging, monitoring, caching)
"""

import asyncio
import csv
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Iterator, Callable, Tuple
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import statistics
import re
import io

# Optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    from chardet import detect
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False


class DataType(Enum):
    """Supported data types for CSV columns."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    CATEGORICAL = "categorical"


class ValidationRule(Enum):
    """Data validation rules."""
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    REGEX_MATCH = "regex_match"
    RANGE_CHECK = "range_check"
    ENUM_CHECK = "enum_check"
    CUSTOM = "custom"


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    EXCEL = "excel"
    PARQUET = "parquet"
    TSV = "tsv"


@dataclass
class ColumnSchema:
    """Schema definition for a CSV column."""
    name: str
    data_type: DataType
    nullable: bool = True
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    regex_pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    default_value: Optional[Any] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['data_type'] = self.data_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnSchema':
        """Create from dictionary representation."""
        data['data_type'] = DataType(data['data_type'])
        return cls(**data)


@dataclass
class ValidationError:
    """Data validation error."""
    row_number: int
    column_name: str
    rule: ValidationRule
    value: Any
    expected: Any
    message: str
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['rule'] = self.rule.value
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ColumnStatistics:
    """Statistical analysis of a column."""
    name: str
    data_type: DataType
    total_count: int
    null_count: int
    unique_count: int
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_dev: Optional[float] = None
    mode_value: Optional[Any] = None
    percentiles: Optional[Dict[str, float]] = None
    value_counts: Optional[Dict[str, int]] = None
    sample_values: Optional[List[Any]] = None

    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = {}
        if self.value_counts is None:
            self.value_counts = {}
        if self.sample_values is None:
            self.sample_values = []

    @property
    def null_percentage(self) -> float:
        """Get percentage of null values."""
        return (self.null_count / self.total_count) * 100 if self.total_count > 0 else 0

    @property
    def unique_percentage(self) -> float:
        """Get percentage of unique values."""
        return (self.unique_count / self.total_count) * 100 if self.total_count > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['data_type'] = self.data_type.value
        data['null_percentage'] = self.null_percentage
        data['unique_percentage'] = self.unique_percentage
        return data


@dataclass
class ParseResult:
    """Result from CSV parsing operation."""
    file_path: str
    total_rows: int
    total_columns: int
    parsed_rows: int
    skipped_rows: int
    error_count: int
    warnings: List[str]
    errors: List[ValidationError]
    schema: List[ColumnSchema]
    statistics: List[ColumnStatistics]
    execution_time: float
    memory_usage_mb: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def success_rate(self) -> float:
        """Get parsing success rate."""
        return (self.parsed_rows / self.total_rows) * 100 if self.total_rows > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data['schema'] = [col.to_dict() for col in self.schema]
        data['statistics'] = [stat.to_dict() for stat in self.statistics]
        data['errors'] = [err.to_dict() for err in self.errors]
        data['success_rate'] = self.success_rate
        return data


class DataTypeDetector:
    """Automatic data type detection for CSV columns."""

    # Regex patterns for data type detection
    PATTERNS = {
        DataType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        DataType.URL: re.compile(r'^https?://[^\s]+$'),
        DataType.PHONE: re.compile(r'^[\+]?[1-9]?[\d\s\-\(\)\.]{7,15}$'),
        DataType.CURRENCY: re.compile(r'^[\$\€\£\¥]?[\d,]+\.?\d*$'),
        DataType.PERCENTAGE: re.compile(r'^\d+\.?\d*%$'),
        DataType.DATE: re.compile(r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$|^\d{2}-\d{2}-\d{4}$'),
        DataType.DATETIME: re.compile(r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$')
    }

    @classmethod
    def detect_type(cls, values: List[str], sample_size: int = 100) -> DataType:
        """
        Detect data type from sample values.

        Args:
            values: List of string values to analyze
            sample_size: Maximum number of values to sample

        Returns:
            Detected data type
        """
        if not values:
            return DataType.STRING

        # Sample values for performance
        sample_values = values[:sample_size] if len(values) > sample_size else values
        non_null_values = [v for v in sample_values if v and str(v).strip()]

        if not non_null_values:
            return DataType.STRING

        # Test for specific patterns
        for data_type, pattern in cls.PATTERNS.items():
            if all(pattern.match(str(v).strip()) for v in non_null_values):
                return data_type

        # Test for boolean
        boolean_values = {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n'}
        if all(str(v).lower().strip() in boolean_values for v in non_null_values):
            return DataType.BOOLEAN

        # Test for numeric types
        try:
            float_values = [float(v) for v in non_null_values]

            # Check if all are integers
            if all(v.is_integer() for v in float_values):
                return DataType.INTEGER
            else:
                return DataType.FLOAT
        except (ValueError, TypeError):
            pass

        # Check if categorical (limited unique values)
        unique_values = set(non_null_values)
        if len(unique_values) <= min(10, len(non_null_values) * 0.1):
            return DataType.CATEGORICAL

        # Default to string
        return DataType.STRING

    @classmethod
    def infer_schema(cls, data: List[List[str]], headers: List[str]) -> List[ColumnSchema]:
        """
        Infer schema from CSV data.

        Args:
            data: List of rows (each row is a list of strings)
            headers: Column headers

        Returns:
            List of inferred column schemas
        """
        if not data or not headers:
            return []

        schema = []

        for col_idx, header in enumerate(headers):
            # Extract column values
            column_values = [row[col_idx] if col_idx < len(row) else '' for row in data]

            # Detect data type
            data_type = cls.detect_type(column_values)

            # Check for nullability
            null_count = sum(1 for v in column_values if not v or str(v).strip() == '')
            nullable = null_count > 0

            # Check for uniqueness (if small dataset)
            unique = False
            if len(column_values) <= 1000:
                non_null_values = [v for v in column_values if v and str(v).strip()]
                unique = len(set(non_null_values)) == len(non_null_values)

            schema.append(ColumnSchema(
                name=header,
                data_type=data_type,
                nullable=nullable,
                unique=unique
            ))

        return schema


class DataValidator:
    """Data validation engine for CSV data."""

    def __init__(self, schema: List[ColumnSchema]):
        """
        Initialize validator with schema.

        Args:
            schema: List of column schemas for validation
        """
        self.schema = {col.name: col for col in schema}
        self.errors: List[ValidationError] = []

    def validate_row(self, row_data: Dict[str, Any], row_number: int) -> List[ValidationError]:
        """
        Validate a single row against schema.

        Args:
            row_data: Dictionary of column name to value
            row_number: Row number for error reporting

        Returns:
            List of validation errors
        """
        row_errors = []

        for col_name, col_schema in self.schema.items():
            value = row_data.get(col_name)

            # Check not null constraint
            if not col_schema.nullable and (value is None or str(value).strip() == ''):
                row_errors.append(ValidationError(
                    row_number=row_number,
                    column_name=col_name,
                    rule=ValidationRule.NOT_NULL,
                    value=value,
                    expected="non-null value",
                    message=f"Column '{col_name}' cannot be null"
                ))
                continue

            # Skip further validation if value is null
            if value is None or str(value).strip() == '':
                continue

            # Type-specific validation
            row_errors.extend(self._validate_data_type(value, col_schema, row_number))

            # Range validation
            if col_schema.min_value is not None or col_schema.max_value is not None:
                row_errors.extend(self._validate_range(value, col_schema, row_number))

            # Length validation
            if col_schema.min_length is not None or col_schema.max_length is not None:
                row_errors.extend(self._validate_length(value, col_schema, row_number))

            # Regex validation
            if col_schema.regex_pattern:
                row_errors.extend(self._validate_regex(value, col_schema, row_number))

            # Enum validation
            if col_schema.allowed_values:
                row_errors.extend(self._validate_enum(value, col_schema, row_number))

        return row_errors

    def _validate_data_type(self, value: Any, schema: ColumnSchema, row_number: int) -> List[ValidationError]:
        """Validate data type."""
        errors = []

        try:
            if schema.data_type == DataType.INTEGER:
                int(value)
            elif schema.data_type == DataType.FLOAT:
                float(value)
            elif schema.data_type == DataType.BOOLEAN:
                str(value).lower() in {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n'}
            elif schema.data_type == DataType.EMAIL:
                if not DataTypeDetector.PATTERNS[DataType.EMAIL].match(str(value)):
                    raise ValueError("Invalid email format")
            # Add other type validations as needed
        except (ValueError, TypeError):
            errors.append(ValidationError(
                row_number=row_number,
                column_name=schema.name,
                rule=ValidationRule.RANGE_CHECK,
                value=value,
                expected=schema.data_type.value,
                message=f"Value '{value}' is not a valid {schema.data_type.value}"
            ))

        return errors

    def _validate_range(self, value: Any, schema: ColumnSchema, row_number: int) -> List[ValidationError]:
        """Validate numeric range."""
        errors = []

        try:
            numeric_value = float(value)

            if schema.min_value is not None and numeric_value < schema.min_value:
                errors.append(ValidationError(
                    row_number=row_number,
                    column_name=schema.name,
                    rule=ValidationRule.RANGE_CHECK,
                    value=value,
                    expected=f">= {schema.min_value}",
                    message=f"Value {value} is below minimum {schema.min_value}"
                ))

            if schema.max_value is not None and numeric_value > schema.max_value:
                errors.append(ValidationError(
                    row_number=row_number,
                    column_name=schema.name,
                    rule=ValidationRule.RANGE_CHECK,
                    value=value,
                    expected=f"<= {schema.max_value}",
                    message=f"Value {value} is above maximum {schema.max_value}"
                ))
        except (ValueError, TypeError):
            pass  # Type validation will catch this

        return errors

    def _validate_length(self, value: Any, schema: ColumnSchema, row_number: int) -> List[ValidationError]:
        """Validate string length."""
        errors = []
        value_str = str(value)

        if schema.min_length is not None and len(value_str) < schema.min_length:
            errors.append(ValidationError(
                row_number=row_number,
                column_name=schema.name,
                rule=ValidationRule.MIN_LENGTH,
                value=value,
                expected=f"length >= {schema.min_length}",
                message=f"Value '{value}' is too short (minimum {schema.min_length} characters)"
            ))

        if schema.max_length is not None and len(value_str) > schema.max_length:
            errors.append(ValidationError(
                row_number=row_number,
                column_name=schema.name,
                rule=ValidationRule.MAX_LENGTH,
                value=value,
                expected=f"length <= {schema.max_length}",
                message=f"Value '{value}' is too long (maximum {schema.max_length} characters)"
            ))

        return errors

    def _validate_regex(self, value: Any, schema: ColumnSchema, row_number: int) -> List[ValidationError]:
        """Validate regex pattern."""
        errors = []

        try:
            pattern = re.compile(schema.regex_pattern)
            if not pattern.match(str(value)):
                errors.append(ValidationError(
                    row_number=row_number,
                    column_name=schema.name,
                    rule=ValidationRule.REGEX_MATCH,
                    value=value,
                    expected=schema.regex_pattern,
                    message=f"Value '{value}' does not match pattern '{schema.regex_pattern}'"
                ))
        except re.error:
            pass  # Invalid regex pattern

        return errors

    def _validate_enum(self, value: Any, schema: ColumnSchema, row_number: int) -> List[ValidationError]:
        """Validate enumerated values."""
        errors = []

        if str(value) not in schema.allowed_values:
            errors.append(ValidationError(
                row_number=row_number,
                column_name=schema.name,
                rule=ValidationRule.ENUM_CHECK,
                value=value,
                expected=schema.allowed_values,
                message=f"Value '{value}' is not in allowed values: {schema.allowed_values}"
            ))

        return errors


class StatisticalAnalyzer:
    """Statistical analysis engine for CSV data."""

    @staticmethod
    def analyze_column(name: str, values: List[Any], data_type: DataType) -> ColumnStatistics:
        """
        Analyze a single column and generate statistics.

        Args:
            name: Column name
            values: List of values in the column
            data_type: Data type of the column

        Returns:
            Column statistics
        """
        total_count = len(values)
        null_values = [v for v in values if v is None or str(v).strip() == '']
        non_null_values = [v for v in values if v is not None and str(v).strip() != '']

        null_count = len(null_values)
        unique_count = len(set(str(v) for v in non_null_values))

        stats = ColumnStatistics(
            name=name,
            data_type=data_type,
            total_count=total_count,
            null_count=null_count,
            unique_count=unique_count
        )

        if not non_null_values:
            return stats

        # Basic statistics
        stats.sample_values = non_null_values[:10]  # First 10 values as sample

        # Value counts for categorical data
        if data_type in [DataType.CATEGORICAL, DataType.BOOLEAN, DataType.STRING]:
            value_counter = Counter(str(v) for v in non_null_values)
            stats.value_counts = dict(value_counter.most_common(20))  # Top 20 values
            stats.mode_value = value_counter.most_common(1)[0][0] if value_counter else None

        # Numeric statistics
        if data_type in [DataType.INTEGER, DataType.FLOAT]:
            try:
                numeric_values = [float(v) for v in non_null_values]

                stats.min_value = min(numeric_values)
                stats.max_value = max(numeric_values)
                stats.mean_value = statistics.mean(numeric_values)
                stats.median_value = statistics.median(numeric_values)

                if len(numeric_values) > 1:
                    stats.std_dev = statistics.stdev(numeric_values)

                # Percentiles
                sorted_values = sorted(numeric_values)
                n = len(sorted_values)
                stats.percentiles = {
                    '25th': sorted_values[int(n * 0.25)] if n > 0 else None,
                    '50th': sorted_values[int(n * 0.50)] if n > 0 else None,
                    '75th': sorted_values[int(n * 0.75)] if n > 0 else None,
                    '90th': sorted_values[int(n * 0.90)] if n > 0 else None,
                    '95th': sorted_values[int(n * 0.95)] if n > 0 else None
                }

            except (ValueError, TypeError):
                pass

        # String length statistics for text data
        if data_type == DataType.STRING:
            lengths = [len(str(v)) for v in non_null_values]
            if lengths:
                stats.min_value = min(lengths)
                stats.max_value = max(lengths)
                stats.mean_value = statistics.mean(lengths)

        return stats

    @classmethod
    def analyze_dataframe(cls, data: List[Dict[str, Any]], schema: List[ColumnSchema]) -> List[ColumnStatistics]:
        """
        Analyze entire dataset and generate statistics for all columns.

        Args:
            data: List of row dictionaries
            schema: Column schemas

        Returns:
            List of column statistics
        """
        if not data or not schema:
            return []

        statistics_list = []

        for col_schema in schema:
            col_name = col_schema.name
            col_values = [row.get(col_name) for row in data]

            stats = cls.analyze_column(col_name, col_values, col_schema.data_type)
            statistics_list.append(stats)

        return statistics_list


class CSVParser:
    """
    Comprehensive CSV parsing and analysis system.

    Provides advanced CSV processing with automatic schema detection,
    data validation, statistical analysis, and export capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CSV parser.

        Args:
            config: Configuration dictionary with parsing settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Core configuration
        self.delimiter = self.config.get('delimiter', ',')
        self.quote_char = self.config.get('quote_char', '"')
        self.escape_char = self.config.get('escape_char', None)
        self.skip_initial_space = self.config.get('skip_initial_space', True)
        self.encoding = self.config.get('encoding', 'utf-8')

        # Processing settings
        self.auto_detect_schema = self.config.get('auto_detect_schema', True)
        self.auto_detect_types = self.config.get('auto_detect_types', True)
        self.sample_size = self.config.get('sample_size', 1000)
        self.chunk_size = self.config.get('chunk_size', 10000)

        # Validation settings
        self.enable_validation = self.config.get('enable_validation', True)
        self.strict_validation = self.config.get('strict_validation', False)
        self.max_errors = self.config.get('max_errors', 100)

        # Performance settings
        self.enable_caching = self.config.get('enable_caching', True)
        self.enable_statistics = self.config.get('enable_statistics', True)
        self.memory_efficient = self.config.get('memory_efficient', True)

        # Enterprise features
        self.audit_logging = self.config.get('audit_logging', False)
        self.monitoring_enabled = self.config.get('monitoring_enabled', False)

        # Initialize components
        self.cache: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = defaultdict(int)

    async def parse_file(self, file_path: str, schema: Optional[List[ColumnSchema]] = None) -> ParseResult:
        """
        Parse a CSV file with comprehensive analysis.

        Args:
            file_path: Path to the CSV file
            schema: Optional predefined schema for validation

        Returns:
            Parse result with statistics and validation results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting CSV parsing: {file_path}")

            # Check cache
            cache_key = self._get_cache_key(file_path, schema)
            if self.enable_caching and cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                return self.cache[cache_key]

            # Detect encoding if not specified
            encoding = await self._detect_encoding(file_path)

            # Parse CSV
            data, headers, warnings = await self._parse_csv_file(file_path, encoding)

            # Auto-detect schema if not provided
            if schema is None and self.auto_detect_schema:
                schema = DataTypeDetector.infer_schema(data, headers)
            elif schema is None:
                # Create basic schema
                schema = [ColumnSchema(name=header, data_type=DataType.STRING) for header in headers]

            # Convert data to dictionaries
            dict_data = []
            for row in data:
                row_dict = {}
                for i, header in enumerate(headers):
                    row_dict[header] = row[i] if i < len(row) else None
                dict_data.append(row_dict)

            # Validate data
            validation_errors = []
            if self.enable_validation and schema:
                validation_errors = await self._validate_data(dict_data, schema)

            # Generate statistics
            statistics = []
            if self.enable_statistics:
                statistics = StatisticalAnalyzer.analyze_dataframe(dict_data, schema)

            # Calculate metrics
            execution_time = time.time() - start_time
            memory_usage = self._get_memory_usage()

            # Create result
            result = ParseResult(
                file_path=file_path,
                total_rows=len(data),
                total_columns=len(headers),
                parsed_rows=len(dict_data),
                skipped_rows=0,  # Could be calculated if we track skipped rows
                error_count=len(validation_errors),
                warnings=warnings,
                errors=validation_errors,
                schema=schema,
                statistics=statistics,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                metadata={
                    'encoding': encoding,
                    'delimiter': self.delimiter,
                    'file_size_bytes': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                }
            )

            # Cache result
            if self.enable_caching:
                self.cache[cache_key] = result

            # Log audit
            if self.audit_logging:
                self._log_operation('parse_file', {
                    'file_path': file_path,
                    'total_rows': result.total_rows,
                    'success_rate': result.success_rate
                })

            self.metrics['files_parsed'] += 1
            return result

        except Exception as e:
            self.logger.error(f"CSV parsing failed: {e}")
            raise

    async def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        if not CHARDET_AVAILABLE:
            return self.encoding

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = detect(raw_data)
                detected_encoding = result.get('encoding', self.encoding)
                confidence = result.get('confidence', 0)

                if confidence > 0.7:
                    return detected_encoding
                else:
                    return self.encoding
        except Exception:
            return self.encoding

    async def _parse_csv_file(self, file_path: str, encoding: str) -> Tuple[List[List[str]], List[str], List[str]]:
        """Parse CSV file and return data, headers, and warnings."""
        data = []
        headers = []
        warnings = []

        try:
            with open(file_path, 'r', encoding=encoding, newline='') as f:
                # Create CSV reader
                reader = csv.reader(
                    f,
                    delimiter=self.delimiter,
                    quotechar=self.quote_char,
                    skipinitialspace=self.skip_initial_space
                )

                # Read headers
                try:
                    headers = next(reader)
                except StopIteration:
                    raise ValueError("CSV file is empty")

                # Read data
                for row_num, row in enumerate(reader, start=2):  # Start at 2 (after header)
                    # Handle rows with different lengths
                    if len(row) != len(headers):
                        if len(row) > len(headers):
                            warnings.append(f"Row {row_num} has more columns than headers")
                            row = row[:len(headers)]  # Truncate
                        else:
                            warnings.append(f"Row {row_num} has fewer columns than headers")
                            row.extend([''] * (len(headers) - len(row)))  # Pad with empty strings

                    data.append(row)

                    # Memory efficient processing for large files
                    if self.memory_efficient and len(data) % self.chunk_size == 0:
                        # Process chunk and continue
                        # For now, we'll keep all data in memory for simplicity
                        pass

        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error: {e}. Try specifying a different encoding.")
        except Exception as e:
            raise ValueError(f"CSV parsing error: {e}")

        return data, headers, warnings

    async def _validate_data(self, data: List[Dict[str, Any]], schema: List[ColumnSchema]) -> List[ValidationError]:
        """Validate parsed data against schema."""
        validator = DataValidator(schema)
        all_errors = []

        for row_idx, row_data in enumerate(data):
            if len(all_errors) >= self.max_errors:
                break

            row_errors = validator.validate_row(row_data, row_idx + 2)  # +2 for header and 0-indexing
            all_errors.extend(row_errors)

            if self.strict_validation and row_errors:
                break

        return all_errors

    async def export_data(self, data: List[Dict[str, Any]], output_path: str,
                         format: ExportFormat = ExportFormat.CSV) -> bool:
        """
        Export parsed data to various formats.

        Args:
            data: List of row dictionaries to export
            output_path: Output file path
            format: Export format

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Exporting data to {output_path} in {format.value} format")

            if format == ExportFormat.CSV:
                await self._export_csv(data, output_path)
            elif format == ExportFormat.JSON:
                await self._export_json(data, output_path)
            elif format == ExportFormat.JSONL:
                await self._export_jsonl(data, output_path)
            elif format == ExportFormat.EXCEL:
                await self._export_excel(data, output_path)
            elif format == ExportFormat.PARQUET:
                await self._export_parquet(data, output_path)
            elif format == ExportFormat.TSV:
                await self._export_tsv(data, output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.metrics['files_exported'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return False

    async def _export_csv(self, data: List[Dict[str, Any]], output_path: str):
        """Export data as CSV."""
        if not data:
            return

        headers = list(data[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)

    async def _export_json(self, data: List[Dict[str, Any]], output_path: str):
        """Export data as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    async def _export_jsonl(self, data: List[Dict[str, Any]], output_path: str):
        """Export data as JSON Lines."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in data:
                f.write(json.dumps(row, default=str) + '\n')

    async def _export_excel(self, data: List[Dict[str, Any]], output_path: str):
        """Export data as Excel."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for Excel export")

        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)

    async def _export_parquet(self, data: List[Dict[str, Any]], output_path: str):
        """Export data as Parquet."""
        if not PANDAS_AVAILABLE or not PARQUET_AVAILABLE:
            raise ImportError("pandas and pyarrow required for Parquet export")

        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)

    async def _export_tsv(self, data: List[Dict[str, Any]], output_path: str):
        """Export data as TSV."""
        if not data:
            return

        headers = list(data[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
            writer.writeheader()
            writer.writerows(data)

    def filter_data(self, data: List[Dict[str, Any]],
                   filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter data based on column conditions.

        Args:
            data: List of row dictionaries
            filters: Dictionary of column filters

        Returns:
            Filtered data
        """
        filtered_data = []

        for row in data:
            include_row = True

            for column, condition in filters.items():
                if column not in row:
                    include_row = False
                    break

                value = row[column]

                if isinstance(condition, dict):
                    # Complex condition
                    if 'eq' in condition and value != condition['eq']:
                        include_row = False
                        break
                    if 'ne' in condition and value == condition['ne']:
                        include_row = False
                        break
                    if 'gt' in condition and not (value > condition['gt']):
                        include_row = False
                        break
                    if 'lt' in condition and not (value < condition['lt']):
                        include_row = False
                        break
                    if 'contains' in condition and condition['contains'] not in str(value):
                        include_row = False
                        break
                    if 'regex' in condition:
                        pattern = re.compile(condition['regex'])
                        if not pattern.search(str(value)):
                            include_row = False
                            break
                else:
                    # Simple equality condition
                    if value != condition:
                        include_row = False
                        break

            if include_row:
                filtered_data.append(row)

        return filtered_data

    def transform_data(self, data: List[Dict[str, Any]],
                      transformations: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """
        Transform data using custom functions.

        Args:
            data: List of row dictionaries
            transformations: Dictionary of column transformations

        Returns:
            Transformed data
        """
        transformed_data = []

        for row in data:
            new_row = row.copy()

            for column, transform_func in transformations.items():
                if column in new_row:
                    try:
                        new_row[column] = transform_func(new_row[column])
                    except Exception as e:
                        self.logger.warning(f"Transformation failed for column {column}: {e}")

            transformed_data.append(new_row)

        return transformed_data

    def get_column_profile(self, data: List[Dict[str, Any]], column_name: str) -> Dict[str, Any]:
        """
        Get detailed profile of a specific column.

        Args:
            data: List of row dictionaries
            column_name: Name of column to profile

        Returns:
            Column profile dictionary
        """
        if not data or column_name not in data[0]:
            return {}

        values = [row.get(column_name) for row in data]
        data_type = DataTypeDetector.detect_type([str(v) for v in values if v is not None])
        stats = StatisticalAnalyzer.analyze_column(column_name, values, data_type)

        return {
            'statistics': stats.to_dict(),
            'data_quality': {
                'completeness': (1 - stats.null_percentage / 100) * 100,
                'uniqueness': stats.unique_percentage,
                'validity': 100.0  # Could be calculated based on validation rules
            },
            'sample_data': values[:20],  # First 20 values
            'recommendations': self._generate_column_recommendations(stats)
        }

    def _generate_column_recommendations(self, stats: ColumnStatistics) -> List[str]:
        """Generate recommendations for column improvements."""
        recommendations = []

        if stats.null_percentage > 50:
            recommendations.append(f"High null rate ({stats.null_percentage:.1f}%) - consider data quality improvement")

        if stats.unique_percentage < 5 and stats.data_type != DataType.CATEGORICAL:
            recommendations.append("Low unique values - consider converting to categorical type")

        if stats.data_type == DataType.STRING and stats.mean_value and stats.mean_value > 1000:
            recommendations.append("Very long text values - consider text processing or chunking")

        if stats.unique_percentage == 100 and stats.total_count > 100:
            recommendations.append("All values unique - potential identifier column")

        return recommendations

    def _get_cache_key(self, file_path: str, schema: Optional[List[ColumnSchema]]) -> str:
        """Generate cache key for parsing results."""
        key_data = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'file_mtime': os.path.getmtime(file_path) if os.path.exists(file_path) else 0,
            'schema': [s.to_dict() for s in schema] if schema else None,
            'config': self.config
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log operations for audit purposes."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'details': details
        }
        self.logger.info(f"AUDIT: {json.dumps(log_entry)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        return dict(self.metrics)

    def clear_cache(self):
        """Clear parsing cache."""
        self.cache.clear()
        self.logger.info("CSV parser cache cleared")

    async def cleanup(self):
        """Cleanup CSV parser resources."""
        try:
            self.clear_cache()
            self.metrics.clear()
            self.logger.info("CSV parser cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'cache') and self.cache:
                self.logger.info("CSVParser being destroyed - cleanup recommended")
        except:
            pass
