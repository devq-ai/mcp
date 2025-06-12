"""
Data Analysis Tool for Agentical

This module provides comprehensive data analysis capabilities including
data processing, statistical analysis, visualization, and reporting
with integration to the Agentical framework.

Features:
- Data loading from multiple sources (CSV, JSON, SQL, APIs)
- Data cleaning and preprocessing
- Statistical analysis and hypothesis testing
- Data visualization with multiple backends
- Machine learning model evaluation
- Automated report generation
- Performance monitoring and caching
- Integration with pandas, numpy, matplotlib, seaborn
"""

import asyncio
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tempfile
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import io
import base64

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError
)
from ...core.logging import log_operation


class DataSource(Enum):
    """Data source types for analysis."""
    CSV = "csv"
    JSON = "json"
    SQL = "sql"
    EXCEL = "excel"
    PARQUET = "parquet"
    API = "api"
    DATAFRAME = "dataframe"


class VisualizationType(Enum):
    """Types of visualizations available."""
    LINE_PLOT = "line"
    BAR_PLOT = "bar"
    SCATTER_PLOT = "scatter"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box"
    HEATMAP = "heatmap"
    CORRELATION_MATRIX = "correlation"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "timeseries"
    INTERACTIVE = "interactive"


class StatisticalTest(Enum):
    """Statistical tests available."""
    T_TEST = "t_test"
    ANOVA = "anova"
    CHI_SQUARE = "chi_square"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    NORMALITY = "normality"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"


class AnalysisResult:
    """Result of data analysis with comprehensive details."""

    def __init__(
        self,
        analysis_id: str,
        analysis_type: str,
        success: bool,
        data_summary: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
        visualizations: Optional[List[Dict[str, Any]]] = None,
        statistical_tests: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.analysis_id = analysis_id
        self.analysis_type = analysis_type
        self.success = success
        self.data_summary = data_summary or {}
        self.results = results or {}
        self.visualizations = visualizations or []
        self.statistical_tests = statistical_tests or []
        self.warnings = warnings or []
        self.execution_time = execution_time
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type,
            "success": self.success,
            "data_summary": self.data_summary,
            "results": self.results,
            "visualizations": self.visualizations,
            "statistical_tests": self.statistical_tests,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class DataAnalyzer:
    """
    Comprehensive data analysis tool with statistical analysis,
    visualization, and reporting capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data analyzer.

        Args:
            config: Configuration for data analysis
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.max_dataframe_rows = self.config.get("max_dataframe_rows", 100000)
        self.max_plot_points = self.config.get("max_plot_points", 10000)
        self.enable_caching = self.config.get("enable_caching", True)
        self.default_visualization = self.config.get("default_visualization", "matplotlib")
        self.output_dir = self.config.get("output_dir", tempfile.gettempdir())

        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Cache for storing analysis results
        self.cache = {}

    @log_operation("data_analysis")
    async def analyze_data(
        self,
        data: Union[pd.DataFrame, str, Dict[str, Any]],
        analysis_type: str = "exploratory",
        source_type: DataSource = DataSource.DATAFRAME,
        target_column: Optional[str] = None,
        features: Optional[List[str]] = None,
        include_visualizations: bool = True,
        include_statistical_tests: bool = True,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive data analysis.

        Args:
            data: Data to analyze (DataFrame, file path, or configuration)
            analysis_type: Type of analysis to perform
            source_type: Type of data source
            target_column: Target variable for supervised analysis
            features: List of feature columns
            include_visualizations: Whether to generate visualizations
            include_statistical_tests: Whether to perform statistical tests
            custom_config: Custom configuration for analysis

        Returns:
            AnalysisResult: Comprehensive analysis results
        """
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now()

        try:
            # Load data if needed
            if not isinstance(data, pd.DataFrame):
                df = await self._load_data(data, source_type)
            else:
                df = data.copy()

            # Validate data
            self._validate_dataframe(df)

            # Perform data summary
            data_summary = self._generate_data_summary(df)

            # Perform specific analysis based on type
            analysis_results = await self._perform_analysis(
                df, analysis_type, target_column, features, custom_config
            )

            # Generate visualizations if requested
            visualizations = []
            if include_visualizations:
                visualizations = await self._generate_visualizations(
                    df, analysis_type, target_column, features
                )

            # Perform statistical tests if requested
            statistical_tests = []
            if include_statistical_tests:
                statistical_tests = await self._perform_statistical_tests(
                    df, analysis_type, target_column, features
                )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            return AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                success=True,
                data_summary=data_summary,
                results=analysis_results,
                visualizations=visualizations,
                statistical_tests=statistical_tests,
                execution_time=execution_time,
                metadata={
                    "data_shape": df.shape,
                    "target_column": target_column,
                    "features": features,
                    "source_type": source_type.value
                }
            )

        except Exception as e:
            self.logger.error(f"Data analysis failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()

            return AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                success=False,
                execution_time=execution_time,
                warnings=[str(e)]
            )

    async def _load_data(
        self,
        data_source: Union[str, Dict[str, Any]],
        source_type: DataSource
    ) -> pd.DataFrame:
        """Load data from various sources."""

        if source_type == DataSource.CSV:
            if isinstance(data_source, str):
                return pd.read_csv(data_source)
            else:
                return pd.read_csv(**data_source)

        elif source_type == DataSource.JSON:
            if isinstance(data_source, str):
                return pd.read_json(data_source)
            else:
                return pd.read_json(**data_source)

        elif source_type == DataSource.EXCEL:
            if isinstance(data_source, str):
                return pd.read_excel(data_source)
            else:
                return pd.read_excel(**data_source)

        elif source_type == DataSource.PARQUET:
            if isinstance(data_source, str):
                return pd.read_parquet(data_source)
            else:
                return pd.read_parquet(**data_source)

        elif source_type == DataSource.SQL:
            # Requires database connection configuration
            if not isinstance(data_source, dict):
                raise ToolValidationError("SQL source requires connection configuration")

            # This would integrate with the database tool
            raise NotImplementedError("SQL data loading requires database tool integration")

        elif source_type == DataSource.API:
            # Requires API configuration
            if not isinstance(data_source, dict):
                raise ToolValidationError("API source requires configuration")

            # This would integrate with web search/fetch tools
            raise NotImplementedError("API data loading requires web tool integration")

        else:
            raise ToolValidationError(f"Unsupported data source type: {source_type}")

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate dataframe for analysis."""
        if df.empty:
            raise ToolValidationError("DataFrame is empty")

        if len(df) > self.max_dataframe_rows:
            raise ToolValidationError(
                f"DataFrame too large: {len(df)} rows > {self.max_dataframe_rows} max"
            )

    def _generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        summary = {
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
            "column_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
        }

        # Descriptive statistics for numeric columns
        if summary["numeric_columns"]:
            summary["descriptive_stats"] = df[summary["numeric_columns"]].describe().to_dict()

        # Value counts for categorical columns (top 10)
        if summary["categorical_columns"]:
            summary["categorical_stats"] = {}
            for col in summary["categorical_columns"]:
                if df[col].nunique() <= 100:  # Only for manageable categories
                    summary["categorical_stats"][col] = df[col].value_counts().head(10).to_dict()

        return summary

    async def _perform_analysis(
        self,
        df: pd.DataFrame,
        analysis_type: str,
        target_column: Optional[str],
        features: Optional[List[str]],
        custom_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform specific type of analysis."""

        if analysis_type == "exploratory":
            return self._exploratory_analysis(df)

        elif analysis_type == "correlation":
            return self._correlation_analysis(df, features)

        elif analysis_type == "regression":
            if not target_column:
                raise ToolValidationError("Regression analysis requires target_column")
            return self._regression_analysis(df, target_column, features)

        elif analysis_type == "classification":
            if not target_column:
                raise ToolValidationError("Classification analysis requires target_column")
            return self._classification_analysis(df, target_column, features)

        elif analysis_type == "time_series":
            return self._time_series_analysis(df, target_column)

        elif analysis_type == "clustering":
            return self._clustering_analysis(df, features)

        elif analysis_type == "custom":
            if not custom_config:
                raise ToolValidationError("Custom analysis requires custom_config")
            return self._custom_analysis(df, custom_config)

        else:
            raise ToolValidationError(f"Unsupported analysis type: {analysis_type}")

    def _exploratory_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform exploratory data analysis."""
        results = {}

        # Data quality assessment
        results["data_quality"] = {
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "unique_values": df.nunique().to_dict()
        }

        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            outliers = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])

            results["outliers"] = outliers

        return results

    def _correlation_analysis(self, df: pd.DataFrame, features: Optional[List[str]]) -> Dict[str, Any]:
        """Perform correlation analysis."""
        numeric_df = df.select_dtypes(include=[np.number])

        if features:
            numeric_df = numeric_df[features]

        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation analysis"}

        correlation_matrix = numeric_df.corr()

        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": self._find_strong_correlations(correlation_matrix),
            "correlation_summary": {
                "mean_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].mean(),
                "max_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].max(),
                "min_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, 1)].min()
            }
        }

    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations in correlation matrix."""
        strong_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corr.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_value
                    })

        return sorted(strong_corr, key=lambda x: abs(x["correlation"]), reverse=True)

    def _regression_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        features: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Perform regression analysis."""
        if target_column not in df.columns:
            raise ToolValidationError(f"Target column '{target_column}' not found")

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in features:
                features.remove(target_column)

        # Basic linear regression using scipy
        results = {}

        for feature in features:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                # Remove NaN values
                clean_data = df[[feature, target_column]].dropna()

                if len(clean_data) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        clean_data[feature], clean_data[target_column]
                    )

                    results[feature] = {
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_value ** 2,
                        "p_value": p_value,
                        "std_error": std_err
                    }

        return results

    def _classification_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        features: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Perform classification analysis."""
        if target_column not in df.columns:
            raise ToolValidationError(f"Target column '{target_column}' not found")

        results = {
            "target_distribution": df[target_column].value_counts().to_dict(),
            "class_balance": df[target_column].value_counts(normalize=True).to_dict()
        }

        # Feature importance analysis (basic correlation with target)
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in features:
                features.remove(target_column)

        if features and pd.api.types.is_numeric_dtype(df[target_column]):
            feature_importance = {}
            for feature in features:
                if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                    clean_data = df[[feature, target_column]].dropna()
                    if len(clean_data) > 1:
                        corr, p_value = stats.pearsonr(clean_data[feature], clean_data[target_column])
                        feature_importance[feature] = {
                            "correlation": corr,
                            "p_value": p_value,
                            "abs_correlation": abs(corr)
                        }

            results["feature_importance"] = feature_importance

        return results

    def _time_series_analysis(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Perform time series analysis."""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        if not datetime_cols:
            # Try to identify datetime columns
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    datetime_cols.append(col)
                    break
                except:
                    continue

        if not datetime_cols:
            return {"error": "No datetime columns found for time series analysis"}

        results = {}
        date_col = datetime_cols[0]

        if target_column and target_column in df.columns:
            # Sort by date
            ts_df = df.sort_values(date_col)

            results["trend_analysis"] = {
                "data_points": len(ts_df),
                "date_range": {
                    "start": ts_df[date_col].min().isoformat() if pd.notna(ts_df[date_col].min()) else None,
                    "end": ts_df[date_col].max().isoformat() if pd.notna(ts_df[date_col].max()) else None
                }
            }

            # Basic statistics
            if pd.api.types.is_numeric_dtype(ts_df[target_column]):
                results["statistics"] = {
                    "mean": ts_df[target_column].mean(),
                    "std": ts_df[target_column].std(),
                    "trend": "increasing" if ts_df[target_column].iloc[-1] > ts_df[target_column].iloc[0] else "decreasing"
                }

        return results

    def _clustering_analysis(self, df: pd.DataFrame, features: Optional[List[str]]) -> Dict[str, Any]:
        """Perform clustering analysis."""
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        if not features:
            return {"error": "No numeric features found for clustering analysis"}

        # Basic clustering analysis without sklearn to avoid dependency
        numeric_df = df[features].select_dtypes(include=[np.number])

        return {
            "feature_summary": numeric_df.describe().to_dict(),
            "feature_correlations": numeric_df.corr().to_dict(),
            "data_shape": numeric_df.shape
        }

    def _custom_analysis(self, df: pd.DataFrame, custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform custom analysis based on configuration."""
        # This would be extended based on specific requirements
        return {"message": "Custom analysis not yet implemented", "config": custom_config}

    async def _generate_visualizations(
        self,
        df: pd.DataFrame,
        analysis_type: str,
        target_column: Optional[str],
        features: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Generate visualizations for the analysis."""
        visualizations = []

        try:
            # Limit data points for performance
            if len(df) > self.max_plot_points:
                plot_df = df.sample(n=self.max_plot_points)
            else:
                plot_df = df

            # Generate different visualizations based on analysis type
            if analysis_type == "exploratory":
                visualizations.extend(self._create_exploratory_plots(plot_df))
            elif analysis_type == "correlation":
                visualizations.extend(self._create_correlation_plots(plot_df, features))
            elif analysis_type in ["regression", "classification"]:
                if target_column:
                    visualizations.extend(self._create_target_plots(plot_df, target_column, features))

        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")

        return visualizations

    def _create_exploratory_plots(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create exploratory data analysis plots."""
        plots = []

        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5 columns

        for col in numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                df[col].hist(bins=30, alpha=0.7, ax=ax)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')

                # Save plot to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                plots.append({
                    "type": "histogram",
                    "title": f"Distribution of {col}",
                    "data": plot_data,
                    "format": "base64_png"
                })

            except Exception as e:
                self.logger.warning(f"Failed to create histogram for {col}: {e}")

        return plots

    def _create_correlation_plots(self, df: pd.DataFrame, features: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Create correlation visualization plots."""
        plots = []

        numeric_df = df.select_dtypes(include=[np.number])
        if features:
            numeric_df = numeric_df[features]

        if len(numeric_df.columns) > 1:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                correlation_matrix = numeric_df.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Matrix')

                # Save plot to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                plots.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "data": plot_data,
                    "format": "base64_png"
                })

            except Exception as e:
                self.logger.warning(f"Failed to create correlation plot: {e}")

        return plots

    def _create_target_plots(
        self,
        df: pd.DataFrame,
        target_column: str,
        features: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Create plots related to target variable."""
        plots = []

        if target_column not in df.columns:
            return plots

        # Target distribution
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            if pd.api.types.is_numeric_dtype(df[target_column]):
                df[target_column].hist(bins=30, alpha=0.7, ax=ax)
                ax.set_title(f'Distribution of {target_column}')
            else:
                df[target_column].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Distribution of {target_column}')
                ax.tick_params(axis='x', rotation=45)

            # Save plot to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            plots.append({
                "type": "target_distribution",
                "title": f"Distribution of {target_column}",
                "data": plot_data,
                "format": "base64_png"
            })

        except Exception as e:
            self.logger.warning(f"Failed to create target distribution plot: {e}")

        return plots

    async def _perform_statistical_tests(
        self,
        df: pd.DataFrame,
        analysis_type: str,
        target_column: Optional[str],
        features: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Perform relevant statistical tests."""
        tests = []

        try:
            # Normality tests for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit tests

            for col in numeric_cols:
                if len(df[col].dropna()) > 3:  # Minimum sample size
                    stat, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))

                    tests.append({
                        "test_type": "normality_shapiro",
                        "column": col,
                        "statistic": stat,
                        "p_value": p_value,
                        "is_normal": p_value > 0.05,
                        "interpretation": "Normal distribution" if p_value > 0.05 else "Non-normal distribution"
                    })

        except Exception as e:
            self.logger.warning(f"Statistical tests failed: {e}")

        return tests

    async def generate_report(
        self,
        analysis_result: AnalysisResult,
        format_type: str = "markdown"
    ) -> str:
        """Generate a comprehensive analysis report."""

        if format_type == "markdown":
            return self._generate_markdown_report(analysis_result)
        elif format_type == "html":
            return self._generate_html_report(analysis_result)
        else:
            raise ToolValidationError(f"Unsupported report format: {format_type}")

    def _generate_markdown_report(self, result: AnalysisResult) -> str:
        """Generate markdown format report."""
        report = f"""# Data Analysis Report

## Analysis Overview
- **Analysis ID**: {result.analysis_id}
- **Analysis Type**: {result.analysis_type}
- **Timestamp**: {result.timestamp}
- **Execution Time**: {result.execution_time:.2f} seconds
- **Success**: {'✅' if result.success else '❌'}

## Data Summary
"""

        if result.data_summary:
            ds = result.data_summary
            report += f"""
- **Shape**: {ds.get('shape', 'N/A')}
- **Memory Usage**: {ds.get('memory_usage', 'N/A')} bytes
- **Numeric Columns**: {len(ds.get('numeric_columns', []))}
- **Categorical Columns**: {len(ds.get('categorical_columns', []))}
- **Missing Values**: {sum(ds.get('missing_values', {}).values())}
"""

        if result.results:
            report += "\n## Analysis Results\n"
            for key, value in result.results.items():
                report += f"- **{key}**: {value}\n"

        if result.statistical_tests:
            report += "\n## Statistical Tests\n"
            for test in result.statistical_tests:
                report += f"- **{test.get('test_type', 'Unknown')}** on {test.get('column', 'N/A')}: p-value = {test.get('p_value', 'N/A')}\n"

        if result.warnings:
            report += "\n## Warnings\n"
            for warning in result.warnings:
                report += f"- ⚠️ {warning}\n"

        return report

    def _generate_html_report(self, result: AnalysisResult) -> str:
        """Generate HTML format report."""
        # Basic HTML template - could be enhanced with proper templating
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #333; border-bottom: 2px solid #ccc; }}
                .section {{ margin: 20px 0; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Analysis Report</h1>
                <p>Analysis ID: {result.analysis_id}</p>
                <p>Type: {result.analysis_type}</p>
                <p>Status: <span class="{'success' if result.success else 'error'}">
                    {'Success' if result.success else 'Failed'}
                </span></p>
            </div>

            <div class="section">
                <h2>Data Summary</h2>
                <!-- Data summary content would go here -->
            </div>

            <div class="section">
                <h2>Analysis Results</h2>
                <!-- Analysis results content would go here -->
            </div>
        </body>
        </html>
        """
        return html

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on data analyzer."""
        health_status = {
            "status": "healthy",
            "configuration": {
                "max_dataframe_rows": self.max_dataframe_rows,
                "max_plot_points": self.max_plot_points,
                "enable_caching": self.enable_caching,
                "default_visualization": self.default_visualization
            },
            "dependencies": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "matplotlib": "available",
                "seaborn": "available"
            }
        }

        # Test basic functionality
        try:
            test_df = pd.DataFrame({
                'x': np.random.randn(100),
                'y': np.random.randn(100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })

            test_result = await self.analyze_data(
                test_df,
                analysis_type="exploratory",
                include_visualizations=False,
                include_statistical_tests=False
            )

            health_status["basic_analysis"] = test_result.success

        except Exception as e:
            health_status["status"] = "degraded"
            health_status["basic_analysis"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating data analyzer
def create_data_analyzer(config: Optional[Dict[str, Any]] = None) -> DataAnalyzer:
    """
    Create a data analyzer with specified configuration.

    Args:
        config: Configuration for data analysis

    Returns:
        DataAnalyzer: Configured data analyzer instance
    """
    return DataAnalyzer(config=config)
