import os
import sys
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from crewai_tools import BaseTool
import io  # To capture print output like df.info()
from typing import Dict, List, Any, Optional, Union
import traceback
from contextlib import redirect_stdout, redirect_stderr

# Configure logging
logger = logging.getLogger('crewai-viz-data-tools')

# Define constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, 'datasets')

# Ensure datasets directory exists
if not os.path.exists(DATASET_FOLDER):
    try:
        os.makedirs(DATASET_FOLDER)
        logger.info(f"Created datasets directory at {DATASET_FOLDER}")
    except Exception as e:
        logger.error(f"Failed to create datasets directory: {e}")

class DatasetLoadingTool(BaseTool):
    """Tool for loading datasets from the datasets folder.

    This tool loads a CSV dataset into a pandas DataFrame and returns it for use
    by other tools or agents in the CrewAI workflow.
    """
    name: str = "Dataset Loader Tool"
    description: str = ("Loads a specified dataset from the '{}/' folder "
                       "into a pandas DataFrame. Input must be the dataset name "
                       "(without .csv extension)." ).format(DATASET_FOLDER)

    def _run(self, dataset_name: str) -> Union[pd.DataFrame, str]:
        """Load a dataset from the datasets folder.

        Args:
            dataset_name: Name of the dataset to load (without .csv extension)

        Returns:
            Either a pandas DataFrame containing the dataset or an error message string
        """
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            error_msg = "Dataset name must be a non-empty string."
            logger.error(error_msg)
            return error_msg

        # Security check to prevent directory traversal
        if '..' in dataset_name or '/' in dataset_name or '\\' in dataset_name:
            error_msg = f"Invalid dataset name format: {dataset_name}"
            logger.error(error_msg)
            return error_msg

        try:
            file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
            if not os.path.exists(file_path):
                error_msg = f"Error: Dataset '{dataset_name}' not found at '{file_path}'."
                logger.warning(error_msg)
                return error_msg

            logger.info(f"Loading dataset: {dataset_name}")
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded dataset '{dataset_name}' with shape {df.shape}")

            # Return the dataframe directly for use by other tools/agents in sequence
            return df

        except Exception as e:
            error_msg = f"Error loading dataset {dataset_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

class DatasetInfoTool(BaseTool):
    """Tool for providing detailed information about a dataset.

    This tool analyzes a dataset and returns comprehensive information about its
    structure, content, and statistics to help agents understand the data.
    """
    name: str = "Dataset Information Tool"
    description: str = ("Provides information about a dataset including column names, "
                       "data types, sample rows, and basic statistics. "
                       "Input must be the dataset name (without .csv extension).")

    def _run(self, dataset_name: str) -> str:
        """Get comprehensive information about a dataset.

        Args:
            dataset_name: Name of the dataset to analyze (without .csv extension)

        Returns:
            A formatted string containing dataset information and statistics
        """
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            error_msg = "Dataset name must be a non-empty string."
            logger.error(error_msg)
            return error_msg

        # Security check to prevent directory traversal
        if '..' in dataset_name or '/' in dataset_name or '\\' in dataset_name:
            error_msg = f"Invalid dataset name format: {dataset_name}"
            logger.error(error_msg)
            return error_msg

        try:
            file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
            if not os.path.exists(file_path):
                error_msg = f"Error: Dataset '{dataset_name}' not found."
                logger.warning(error_msg)
                return error_msg

            logger.info(f"Analyzing dataset: {dataset_name}")
            df = pd.read_csv(file_path)

            # Use io.StringIO to capture pandas .info() output
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()

            # Generate correlation information for numerical columns
            correlation_info = ""
            try:
                numerical_cols = df.select_dtypes(include=['number']).columns
                if len(numerical_cols) > 1:
                    correlation_matrix = df[numerical_cols].corr()
                    correlation_info = f"\n\nCorrelation Matrix:\n{correlation_matrix.round(2).to_string()}"
            except Exception as corr_err:
                logger.warning(f"Could not generate correlation matrix: {corr_err}")
                correlation_info = "\n\nCorrelation Matrix: Could not be generated"

            # Compile the complete dataset information
            result = (f"Dataset: {dataset_name}\n"
                     f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
                     f"First 3 rows:\n{df.head(3).to_string()}\n\n"
                     f"Column Names: {df.columns.tolist()}\n\n"
                     f"Data Info:\n{info_str}\n"
                     f"Basic Statistics (numerical):\n{df.describe().to_string()}\n\n"
                     f"Basic Statistics (all):\n{df.describe(include='all').to_string()}"
                     f"{correlation_info}")

            logger.info(f"Successfully analyzed dataset '{dataset_name}'")
            return result

        except Exception as e:
            error_msg = f"Error analyzing dataset {dataset_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

class PythonCodeExecutionTool(BaseTool):
    """Tool for executing Python code to generate visualizations.

    This tool executes Python code provided by the Visualization Coder agent
    and returns the results, including any generated Plotly or ECharts visualizations.
    """
    name: str = "Python Code Execution Tool"
    description: str = ("Executes Python code to generate visualizations. "
                       "The code should use pandas, plotly, and other libraries "
                       "to process data and create visualizations. "
                       "Input must be valid Python code as a string.")

    def _run(self, code: str, dataset_name: str = None) -> str:
        """Execute Python code and return the results.

        Args:
            code: Python code to execute
            dataset_name: Optional name of the dataset to load

        Returns:
            A string containing the execution results, including any generated visualizations
        """
        if not isinstance(code, str) or not code.strip():
            error_msg = "Code must be a non-empty string."
            logger.error(error_msg)
            return error_msg

        # Create a safe execution environment
        local_vars = {
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'plt': None,  # Will be replaced with matplotlib.pyplot if needed
            'json': json,
            'df': None,  # Will be populated with the dataset if provided
            'plotly_figs': [],  # To store generated Plotly figures
            'echarts_configs': []  # To store generated ECharts configurations
        }

        # Common column name mappings to handle mismatches
        column_mappings = {
            'sepal length': 'sepal_length',
            'sepal length (cm)': 'sepal_length',
            'sepal_length (cm)': 'sepal_length',
            'sepal width': 'sepal_width',
            'sepal width (cm)': 'sepal_width',
            'sepal_width (cm)': 'sepal_width',
            'petal length': 'petal_length',
            'petal length (cm)': 'petal_length',
            'petal_length (cm)': 'petal_length',
            'petal width': 'petal_width',
            'petal width (cm)': 'petal_width',
            'petal_width (cm)': 'petal_width'
        }

        # Load the dataset if provided
        if dataset_name:
            try:
                df = load_dataframe(dataset_name)
                if df is not None:
                    # Create a copy of the dataframe with additional column aliases
                    df_with_aliases = df.copy()

                    # Add column aliases to handle common mismatches
                    for alias, original in column_mappings.items():
                        if original in df.columns and alias not in df.columns:
                            df_with_aliases[alias] = df[original]

                    local_vars['df'] = df_with_aliases
                    logger.info(f"Loaded dataset '{dataset_name}' for code execution with column aliases")
                else:
                    return f"Error: Could not load dataset '{dataset_name}'"
            except Exception as e:
                logger.error(f"Error loading dataset for code execution: {e}")
                return f"Error loading dataset: {str(e)}"

        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Try to fix common errors in the code
            fixed_code = code

            # Fix common column name issues
            for alias, original in column_mappings.items():
                # Look for patterns like px.box(df, x='species', y='sepal length (cm)')
                fixed_code = re.sub(
                    r"(['\"])(" + re.escape(alias) + r")(['\"])",
                    r"\1" + alias + r"\3",
                    fixed_code
                )

            # Add helper functions to the execution environment
            helper_code = """
# Helper function to save Plotly figure for frontend
def save_plotly_fig(fig):
    if fig is not None:
        import os
        import json
        import uuid
        import plotly

        # Create a unique ID for this figure
        fig_id = str(uuid.uuid4())

        # Save the figure as JSON in the temp folder
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_visualizations')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the figure as JSON
        fig_path = os.path.join(temp_dir, f'plotly_{fig_id}.json')
        with open(fig_path, 'w') as f:
            f.write(json.dumps(fig.to_dict()))

        # Also add to the in-memory list
        plotly_figs.append(fig)

        # Return the figure ID for reference
        return fig_id
    return None

# Helper function to save ECharts configuration for frontend
def save_echarts_config(config):
    if config is not None:
        import os
        import json
        import uuid

        # Create a unique ID for this config
        config_id = str(uuid.uuid4())

        # Save the config as JSON in the temp folder
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_visualizations')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the config as JSON
        config_path = os.path.join(temp_dir, f'echarts_{config_id}.json')
        with open(config_path, 'w') as f:
            f.write(json.dumps(config))

        # Also add to the in-memory list
        echarts_configs.append(config)

        # Return the config ID for reference
        return config_id
    return None

# Auto-generate basic Plotly figures if none were created
def ensure_plotly_output():
    if len(plotly_figs) == 0 and df is not None:
        try:
            # Get column types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

            # Generate visualizations based on available column types
            visualizations_created = 0

            # 1. If we have numeric columns, create a correlation heatmap
            if len(numeric_cols) >= 2:
                corr_df = df[numeric_cols].corr()
                fig = px.imshow(corr_df,
                               title='Correlation Heatmap',
                               labels=dict(color="Correlation"),
                               color_continuous_scale='RdBu_r')
                save_plotly_fig(fig)
                visualizations_created += 1

            # 2. If we have at least one numeric and one categorical column, create a box plot
            if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                # Use the first categorical column and first numeric column
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]

                # Only use categorical columns with reasonable number of categories
                if df[cat_col].nunique() <= 10:
                    fig = px.box(df, x=cat_col, y=num_col,
                                title=f'Distribution of {num_col} by {cat_col}')
                    save_plotly_fig(fig)
                    visualizations_created += 1

            # 3. If we have at least two numeric columns, create a scatter plot
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               title=f'{numeric_cols[1]} vs {numeric_cols[0]}',
                               opacity=0.7)
                save_plotly_fig(fig)
                visualizations_created += 1

            # 4. For each numeric column, create a histogram
            for i, col in enumerate(numeric_cols[:2]):  # Limit to first 2 numeric columns
                fig = px.histogram(df, x=col,
                                  title=f'Distribution of {col}',
                                  opacity=0.7)
                save_plotly_fig(fig)
                visualizations_created += 1

            # 5. If we have a datetime column and a numeric column, create a line chart
            if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
                fig = px.line(df, x=datetime_cols[0], y=numeric_cols[0],
                             title=f'{numeric_cols[0]} Over Time')
                save_plotly_fig(fig)
                visualizations_created += 1

            # If no visualizations were created, create a simple table view
            if visualizations_created == 0 and len(df) > 0:
                # Create a table view of the data
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(df.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[df[col] for col in df.columns],
                              fill_color='lavender',
                              align='left'))
                ])
                fig.update_layout(title='Data Table View')
                save_plotly_fig(fig)

        except Exception as e:
            print(f"Could not auto-generate Plotly figure: {e}")
            # Last resort - create a simple table view
            try:
                # Create a table view of the first 10 rows
                sample_df = df.head(10)
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(sample_df.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[sample_df[col] for col in sample_df.columns],
                              fill_color='lavender',
                              align='left'))
                ])
                fig.update_layout(title='Data Sample (First 10 Rows)')
                save_plotly_fig(fig)
            except Exception as e2:
                print(f"Failed to create table view: {e2}")
"""
            # Combine helper code with user code and add auto-generation at the end
            full_code = helper_code + "\n" + fixed_code + "\n\n# Ensure we have at least one visualization\nensure_plotly_output()"

            # Print the code being executed (for debugging)
            print("Executing code with dataset:", dataset_name)

            # Execute the code
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(full_code, {}, local_vars)

            # Get the execution results
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()

            # Process any generated Plotly figures
            plotly_configs = []
            for fig in local_vars.get('plotly_figs', []):
                try:
                    # Convert Plotly figure to JSON configuration
                    fig_json = fig.to_json()
                    fig_dict = json.loads(fig_json)
                    plotly_configs.append({
                        'data': fig_dict.get('data', []),
                        'layout': fig_dict.get('layout', {})
                    })
                except Exception as e:
                    logger.error(f"Error converting Plotly figure to JSON: {e}")

            # Process any generated ECharts configurations
            echarts_configs = local_vars.get('echarts_configs', [])

            # Compile the results
            result = {
                'stdout': stdout,
                'stderr': stderr,
                'plotly_configs': plotly_configs,
                'echarts_configs': echarts_configs
            }

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            traceback.print_exc()
            return f"Error executing code: {str(e)}\n{traceback.format_exc()}"

# Initialize tools for export
dataset_loader_tool = DatasetLoadingTool()
dataset_info_tool = DatasetInfoTool()
python_code_execution_tool = PythonCodeExecutionTool()


def load_dataframe(dataset_name: str) -> Optional[pd.DataFrame]:
    """Helper function to load a DataFrame directly (bypassing the tool interface).

    This function is used internally by the crew definition to load datasets
    without going through the CrewAI tool interface.

    Args:
        dataset_name: Name of the dataset to load (without .csv extension)

    Returns:
        A pandas DataFrame if successful, None otherwise
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        logger.error("Dataset name must be a non-empty string.")
        return None

    # Security check to prevent directory traversal
    if '..' in dataset_name or '/' in dataset_name or '\\' in dataset_name:
        logger.error(f"Invalid dataset name format: {dataset_name}")
        return None

    try:
        file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"Error: Dataset '{dataset_name}' not found.")
            return None

        logger.info(f"Loading dataframe: {dataset_name}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataframe '{dataset_name}' with shape {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading dataframe {dataset_name}: {str(e)}")
        return None


def get_dataset_metadata(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Get metadata about a dataset without loading the full DataFrame.

    This function provides quick access to dataset metadata like number of rows,
    columns, and column types without loading the entire dataset into memory.

    Args:
        dataset_name: Name of the dataset (without .csv extension)

    Returns:
        Dictionary with dataset metadata if successful, None otherwise
    """
    try:
        file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            logger.warning(f"Dataset '{dataset_name}' not found.")
            return None

        # Read just the header and first few rows to get metadata
        df_sample = pd.read_csv(file_path, nrows=5)

        # Get full row count without loading entire file
        with open(file_path, 'r') as f:
            row_count = sum(1 for _ in f) - 1  # Subtract 1 for header

        metadata = {
            "name": dataset_name,
            "path": file_path,
            "rows": row_count,
            "columns": len(df_sample.columns),
            "column_names": df_sample.columns.tolist(),
            "column_types": {col: str(dtype) for col, dtype in df_sample.dtypes.items()},
            "sample_rows": df_sample.to_dict(orient='records')
        }

        logger.info(f"Retrieved metadata for dataset '{dataset_name}'")
        return metadata

    except Exception as e:
        logger.error(f"Error getting metadata for {dataset_name}: {str(e)}")
        return None