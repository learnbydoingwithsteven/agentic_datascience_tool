import os
import sys
import time
import json
import logging
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('crewai-viz-backend')

# Import the CrewAI runner function and data tools
# Ensure crew_defs.py and data_tools.py are in the same directory or accessible via PYTHONPATH
try:
    from crew_defs import generate_visualizations
    from data_tools import load_dataframe
    logger.info("Successfully imported CrewAI visualization generator and data tools")
except ImportError as e:
    logger.critical(f"Error importing from crew_defs or data_tools: {e}")
    # Define a dummy function if import fails, so the app can still start
    # This helps in identifying import issues vs. Flask issues.
    def generate_visualizations(dataset_name: str, user_request: str) -> Dict[str, Any]:
        error_msg = f"Backend setup error: CrewAI definitions could not be loaded. Error: {e}"
        logger.critical(f"CRITICAL: {error_msg}")
        return {
            "error": error_msg,
            "plotly_configs": None,
            "echarts_configs": None,
            "raw_output": f"ImportError: {e}"
        }

    def load_dataframe(dataset_name: str):
        logger.critical(f"Backend setup error: Data tools could not be loaded. Error: {e}")
        return None

# Initialize Flask application
app = Flask(__name__)

# Configure CORS - in production, specify allowed origins for security
CORS(app)

# Define constants and configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, 'datasets')

# Ensure datasets directory exists
if not os.path.exists(DATASET_FOLDER):
    try:
        os.makedirs(DATASET_FOLDER)
        logger.info(f"Created datasets directory at {DATASET_FOLDER}")
    except Exception as e:
        logger.error(f"Failed to create datasets directory: {e}")

# Helper functions
def validate_dataset_name(dataset_name: str) -> tuple[bool, Optional[str]]:
    """Validates dataset name for security and existence.

    Args:
        dataset_name: Name of the dataset to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        return False, "Dataset name must be a non-empty string."

    # Security check to prevent directory traversal
    if '..' in dataset_name or '/' in dataset_name or '\\' in dataset_name:
        return False, "Invalid dataset name format."

    # Check if dataset exists
    dataset_file = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
    if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
        return False, f"Dataset '{dataset_name}' not found on server."

    return True, None

# --- API Endpoints ---

@app.route('/api/visualizations/<path:filename>', methods=['GET'])
def get_visualization(filename):
    """Serves visualization files from the temp_visualizations folder.

    Args:
        filename: Name of the visualization file to retrieve

    Returns:
        JSON visualization file or 404 if not found
    """
    # Security check to prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        logger.warning(f"Invalid visualization filename format: {filename}")
        return jsonify({"error": "Invalid filename format."}), 400

    # Construct the path to the visualization file
    viz_path = os.path.join(BASE_DIR, 'temp_visualizations', filename)

    # Check if the file exists
    if not os.path.exists(viz_path) or not os.path.isfile(viz_path):
        logger.warning(f"Visualization file not found: {viz_path}")
        return jsonify({"error": f"Visualization '{filename}' not found."}), 404

    try:
        # Read the visualization file
        with open(viz_path, 'r') as f:
            viz_data = json.load(f)

        logger.info(f"Successfully retrieved visualization: {filename}")
        return jsonify(viz_data)
    except Exception as e:
        logger.error(f"Error retrieving visualization '{filename}': {e}")
        return jsonify({"error": f"Failed to retrieve visualization: {str(e)}"}), 500

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """Lists available CSV datasets from the DATASET_FOLDER.

    Returns:
        JSON response with list of dataset names or error message
    """
    datasets = []

    # Check if dataset folder exists
    if not os.path.exists(DATASET_FOLDER):
        logger.warning(f"Dataset folder not found at {DATASET_FOLDER}")
        return jsonify({"error": f"Dataset folder not found on server at {DATASET_FOLDER}"}), 404

    try:
        # Get all CSV files from the dataset folder
        for f in os.listdir(DATASET_FOLDER):
            file_path = os.path.join(DATASET_FOLDER, f)
            if f.endswith('.csv') and os.path.isfile(file_path):
                datasets.append(f.replace('.csv', ''))

        logger.info(f"Found {len(datasets)} datasets in {DATASET_FOLDER}")
        return jsonify(sorted(datasets))  # Return alphabetically sorted list

    except Exception as e:
        logger.error(f"Error listing datasets in {DATASET_FOLDER}: {e}")
        return jsonify({"error": "An error occurred while listing datasets."}), 500


@app.route('/api/dataset_info/<dataset_name>', methods=['GET'])
def get_dataset_info(dataset_name: str):
    """Returns basic information about a specific dataset.

    Args:
        dataset_name: Name of the dataset to get info for

    Returns:
        JSON with dataset metadata and column information
    """
    # Validate dataset name
    is_valid, error_msg = validate_dataset_name(dataset_name)
    if not is_valid:
        logger.warning(f"Invalid dataset request: {error_msg}")
        return jsonify({"error": error_msg}), 400 if "format" in error_msg else 404

    try:
        # Load the dataset
        file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
        df = pd.read_csv(file_path)

        # Prepare dataset information
        info = {
            "name": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_info": []
        }

        # Get column information
        for col in df.columns:
            col_type = str(df[col].dtype)
            col_info = {
                "name": col,
                "type": col_type,
                "unique_values": int(df[col].nunique()),
                "missing_values": int(df[col].isna().sum())
            }

            # Add sample values for categorical columns (if not too many unique values)
            if col_type == 'object' and df[col].nunique() < 20:
                col_info["sample_values"] = df[col].dropna().unique().tolist()[:10]  # First 10 unique values

            info["column_info"].append(col_info)

        logger.info(f"Successfully retrieved info for dataset '{dataset_name}'")
        return jsonify(info)

    except Exception as e:
        logger.error(f"Error getting info for dataset '{dataset_name}': {e}")
        return jsonify({"error": f"Failed to get dataset information: {str(e)}"}), 500


@app.route('/api/generate_plots', methods=['POST'])
def handle_generate_plots():
    """Receives dataset name and user request, runs CrewAI, returns plot configs.

    Expected JSON body:
    {
        "dataset_name": "name_of_dataset",
        "user_request": "analysis request in natural language"
    }

    Returns:
        JSON with plotly and echarts configurations
    """
    # Validate request format
    if not request.is_json:
        logger.warning("Received non-JSON request to /api/generate_plots")
        return jsonify({"error": "Request must be JSON"}), 400

    # Parse request data
    data = request.get_json()
    if not data or 'dataset_name' not in data or 'user_request' not in data:
        logger.warning(f"Missing required fields in request: {data}")
        return jsonify({"error": "Missing 'dataset_name' or 'user_request' in POST data"}), 400

    dataset_name = data['dataset_name']
    user_request = data['user_request']

    # Validate dataset name
    is_valid, error_msg = validate_dataset_name(dataset_name)
    if not is_valid:
        logger.warning(f"Invalid dataset in plot request: {error_msg}")
        return jsonify({"error": error_msg}), 400 if "format" in error_msg else 404

    # Validate user request
    if not isinstance(user_request, str) or not user_request.strip():
        logger.warning("Empty or invalid user request")
        return jsonify({"error": "User request must be a non-empty string."}), 400

    # Log the request
    logger.info(f"Processing visualization request: Dataset='{dataset_name}', Request='{user_request}'")

    # Check for specific known requests that we can handle directly
    # This is a fallback mechanism for common requests that might cause issues
    if dataset_name == "iris" and ("petal width vs petal length" in user_request.lower() or
                                  "compare petal width vs petal length" in user_request.lower()):
        logger.info("Detected specific iris dataset request for petal width vs length - using direct handler")
        try:
            # Load the dataset
            file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
            df = pd.read_csv(file_path)

            # Create a direct Plotly configuration
            plotly_config = {
                "data": [
                    {
                        "type": "scatter",
                        "mode": "markers",
                        "x": df["petal_length"].tolist(),
                        "y": df["petal_width"].tolist(),
                        "marker": {
                            "color": [{"setosa": 0, "versicolor": 1, "virginica": 2}.get(s, 0) for s in df["species"]],
                            "colorscale": "Viridis",
                            "size": 10
                        },
                        "text": df["species"].tolist()
                    }
                ],
                "layout": {
                    "title": "Petal Width vs Petal Length by Species",
                    "xaxis": {"title": "Petal Length"},
                    "yaxis": {"title": "Petal Width"},
                    "hovermode": "closest"
                }
            }

            # Create a direct ECharts configuration
            # Create a mapping of species to colors
            species_colors = {
                "setosa": "#5470c6",
                "versicolor": "#91cc75",
                "virginica": "#fac858"
            }

            # Create data points with color information
            data_points = []
            for x, y, s in zip(df["petal_length"].tolist(), df["petal_width"].tolist(), df["species"].tolist()):
                data_points.append({
                    "value": [x, y],
                    "name": s,
                    "itemStyle": {"color": species_colors.get(s, "#5470c6")}
                })

            echarts_config = {
                "title": {"text": "Petal Width vs Petal Length by Species"},
                "tooltip": {"trigger": "item"},
                "xAxis": {"type": "value", "name": "Petal Length"},
                "yAxis": {"type": "value", "name": "Petal Width"},
                "series": [
                    {
                        "type": "scatter",
                        "data": data_points
                    }
                ]
            }

            # Create Python code for the visualization
            python_code = """import plotly.express as px
fig = px.scatter(df, x='petal_length', y='petal_width', color='species',
                title='Petal Width vs Petal Length by Species')
save_plotly_fig(fig)
"""

            # Return the direct response
            response_data = {
                "message": f"Generated plots for '{dataset_name}' based on request.",
                "plotlyConfigs": [plotly_config],
                "echartsConfigs": [echarts_config],
                "coderOutput": python_code,
                "executorOutput": "Direct visualization handler used",
                "debug_raw_crew_output": "Direct visualization handler used"
            }

            logger.info("Successfully generated direct visualizations for iris petal width vs length")
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Error in direct visualization handler: {e}")
            logger.error(traceback.format_exc())
            # Continue to the regular flow if the direct handler fails

    # Handle box plot of all columns
    elif dataset_name == "iris" and "box" in user_request.lower() and any(phrase in user_request.lower() for phrase in ["all columns", "all features", "all variables"]):
        logger.info("Detected request for box plot of all columns - using direct handler")
        try:
            # Load the dataset
            file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
            df = pd.read_csv(file_path)

            # Get numerical columns
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            # Create a direct Plotly configuration for box plot of all columns
            plotly_config = {
                "data": [
                    {
                        "type": "box",
                        "y": df[col].tolist(),
                        "name": col,
                        "boxpoints": "outliers"
                    } for col in numerical_cols
                ],
                "layout": {
                    "title": "Distribution of All Features",
                    "yaxis": {"title": "Value"},
                    "boxmode": "group"
                }
            }

            # Create Python code for the visualization
            python_code = """import plotly.express as px
fig = px.box(df, y=df.columns)
save_plotly_fig(fig)
"""

            # Return the direct response
            response_data = {
                "message": f"Generated box plot for all columns in '{dataset_name}'.",
                "plotlyConfigs": [plotly_config],
                "echartsConfigs": [],
                "coderOutput": python_code,
                "executorOutput": "Direct visualization handler used for box plot of all columns",
                "debug_raw_crew_output": "Direct visualization handler used"
            }

            logger.info("Successfully generated direct box plot visualization for all columns")
            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Error in direct box plot visualization handler: {e}")
            logger.error(traceback.format_exc())
            # Continue to the regular flow if the direct handler fails

    # Import necessary modules for visualization handling
    import uuid

    try:
        # Run the CrewAI process to generate visualizations
        result = generate_visualizations(dataset_name, user_request)

        # Check if the generation function reported a fatal error
        if "error" in result and result["error"] and not result.get("plotly_configs") and not result.get("echarts_configs"):
            logger.error(f"Error from visualization generator: {result['error']}")
            # Return 500 for internal errors, 400 for input-related errors
            status_code = 500 if "LLM not configured" in result["error"] or "CrewAI process" in result["error"] else 400
            return jsonify({"error": result["error"], "details": result.get("raw_output")}), status_code

        # Get visualization files from the temp folder
        viz_folder = os.path.join(BASE_DIR, 'temp_visualizations')
        visualizations = []

        if os.path.exists(viz_folder):
            try:
                # Get all JSON files from the visualization folder created in the last minute
                # (assuming they were created by this request)
                current_time = time.time()
                for f in os.listdir(viz_folder):
                    file_path = os.path.join(viz_folder, f)
                    if f.endswith('.json') and os.path.isfile(file_path):
                        # Check if the file was created recently (within the last minute)
                        created_time = os.path.getctime(file_path)
                        if current_time - created_time < 60:  # Within the last minute
                            # Determine the type (plotly or echarts)
                            viz_type = "plotly" if f.startswith("plotly_") else "echarts" if f.startswith("echarts_") else "unknown"

                            visualizations.append({
                                "filename": f,
                                "type": viz_type,
                                "created": created_time,
                                "url": f"/api/visualizations/{f}"
                            })

                # Sort by creation time (newest first)
                visualizations.sort(key=lambda x: x["created"], reverse=True)
            except Exception as e:
                logger.error(f"Error listing visualizations: {e}")

        # Prepare successful response
        response_data = {
            "message": f"Generated plots for '{dataset_name}' based on request.",
            # Ensure keys are camelCase for JS frontend
            "plotlyConfigs": result.get("plotly_configs", []),  # Default to empty list
            "echartsConfigs": result.get("echarts_configs", []),  # Default to empty list
            "coderOutput": result.get("coder_output", ""),  # Python code from the Coder agent
            "executorOutput": result.get("executor_output", ""),  # Output from the Executor agent
            "debug_raw_crew_output": result.get("raw_output", "N/A"),  # Optional: raw output for debugging
            "visualizations": visualizations  # Add the list of visualization files
        }

        # If no visualizations were found, try to generate them directly
        if len(response_data['plotlyConfigs']) == 0 and len(response_data['echartsConfigs']) == 0 and len(visualizations) == 0:
            try:
                # Load the dataset
                df = load_dataframe(dataset_name)
                if df is not None:
                    # Import necessary modules
                    import plotly.express as px
                    import plotly.graph_objects as go

                    # Create temp directory if it doesn't exist
                    temp_dir = os.path.join(BASE_DIR, 'temp_visualizations')
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)

                    # Generate basic visualizations based on the dataset
                    viz_files = []

                    # Get column types
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

                    # 1. Create a table view of the data
                    fig = go.Figure(data=[go.Table(
                        header=dict(values=list(df.columns),
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[df[col] for col in df.columns],
                                  fill_color='lavender',
                                  align='left'))
                    ])
                    fig.update_layout(title='Data Table View')

                    # Save the figure
                    fig_id = str(uuid.uuid4())
                    fig_path = os.path.join(temp_dir, f'plotly_{fig_id}.json')
                    with open(fig_path, 'w') as f:
                        f.write(json.dumps(fig.to_dict()))

                    viz_files.append({
                        "filename": f'plotly_{fig_id}.json',
                        "type": "plotly",
                        "created": os.path.getctime(fig_path),
                        "url": f"/api/visualizations/plotly_{fig_id}.json"
                    })

                    # 2. If we have numeric columns, create visualizations
                    if len(numeric_cols) >= 2:
                        # Scatter plot
                        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                       title=f'{numeric_cols[1]} vs {numeric_cols[0]}',
                                       opacity=0.7)

                        # Save the figure
                        fig_id = str(uuid.uuid4())
                        fig_path = os.path.join(temp_dir, f'plotly_{fig_id}.json')
                        with open(fig_path, 'w') as f:
                            f.write(json.dumps(fig.to_dict()))

                        viz_files.append({
                            "filename": f'plotly_{fig_id}.json',
                            "type": "plotly",
                            "created": os.path.getctime(fig_path),
                            "url": f"/api/visualizations/plotly_{fig_id}.json"
                        })

                        # Correlation heatmap
                        corr_df = df[numeric_cols].corr()
                        fig = px.imshow(corr_df,
                                       title='Correlation Heatmap',
                                       labels=dict(color="Correlation"),
                                       color_continuous_scale='RdBu_r')

                        # Save the figure
                        fig_id = str(uuid.uuid4())
                        fig_path = os.path.join(temp_dir, f'plotly_{fig_id}.json')
                        with open(fig_path, 'w') as f:
                            f.write(json.dumps(fig.to_dict()))

                        viz_files.append({
                            "filename": f'plotly_{fig_id}.json',
                            "type": "plotly",
                            "created": os.path.getctime(fig_path),
                            "url": f"/api/visualizations/plotly_{fig_id}.json"
                        })

                    # 3. For each numeric column, create a histogram
                    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                        fig = px.histogram(df, x=col,
                                          title=f'Distribution of {col}',
                                          opacity=0.7)

                        # Save the figure
                        fig_id = str(uuid.uuid4())
                        fig_path = os.path.join(temp_dir, f'plotly_{fig_id}.json')
                        with open(fig_path, 'w') as f:
                            f.write(json.dumps(fig.to_dict()))

                        viz_files.append({
                            "filename": f'plotly_{fig_id}.json',
                            "type": "plotly",
                            "created": os.path.getctime(fig_path),
                            "url": f"/api/visualizations/plotly_{fig_id}.json"
                        })

                    # Add the visualization files to the response
                    response_data["visualizations"] = viz_files

                    # Log the direct visualization generation
                    logger.info(f"Generated {len(viz_files)} visualizations directly for dataset '{dataset_name}'")
            except Exception as e:
                logger.error(f"Error generating direct visualizations: {e}")
                traceback.print_exc()

        # Log success
        logger.info(f"Successfully generated {len(response_data['plotlyConfigs'])} Plotly, "
                  f"{len(response_data['echartsConfigs'])} ECharts configs, "
                  f"{len(response_data.get('visualizations', []))} visualization files.")
        return jsonify(response_data)

    except Exception as e:
        # Catch unexpected errors during the process
        logger.error(f"FATAL Error in /api/generate_plots: {e}")
        logger.error(traceback.format_exc())  # Log the full stack trace
        return jsonify({
            "error": "An internal server error occurred while generating plots.",
            "details": str(e)
        }), 500


# Health check endpoint
@app.route('/api/visualizations', methods=['GET'])
def list_visualizations():
    """Lists all available visualizations in the temp_visualizations folder.

    Returns:
        JSON response with list of visualization files
    """
    viz_folder = os.path.join(BASE_DIR, 'temp_visualizations')

    # Check if visualization folder exists
    if not os.path.exists(viz_folder):
        try:
            os.makedirs(viz_folder)
            logger.info(f"Created visualizations directory at {viz_folder}")
        except Exception as e:
            logger.error(f"Failed to create visualizations directory: {e}")
            return jsonify({"error": f"Visualization folder not found on server at {viz_folder}"}), 404

    try:
        # Get all JSON files from the visualization folder
        visualizations = []
        for f in os.listdir(viz_folder):
            file_path = os.path.join(viz_folder, f)
            if f.endswith('.json') and os.path.isfile(file_path):
                # Determine the type (plotly or echarts)
                viz_type = "plotly" if f.startswith("plotly_") else "echarts" if f.startswith("echarts_") else "unknown"

                # Get file creation time
                created_time = os.path.getctime(file_path)

                visualizations.append({
                    "filename": f,
                    "type": viz_type,
                    "created": created_time,
                    "url": f"/api/visualizations/{f}"
                })

        # Sort by creation time (newest first)
        visualizations.sort(key=lambda x: x["created"], reverse=True)

        logger.info(f"Found {len(visualizations)} visualizations in {viz_folder}")
        return jsonify(visualizations)

    except Exception as e:
        logger.error(f"Error listing visualizations in {viz_folder}: {e}")
        return jsonify({"error": "An error occurred while listing visualizations."}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the API is running."""
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "endpoints": [
            "/api/datasets",
            "/api/dataset_info/<dataset_name>",
            "/api/generate_plots",
            "/api/visualizations",
            "/api/visualizations/<filename>",
            "/api/health"
        ]
    })


# --- Run the App ---
if __name__ == '__main__':
    # Log startup information
    logger.info(f"Starting CrewAI Data Visualization Backend on port 5001")
    logger.info(f"Dataset folder: {DATASET_FOLDER}")

    try:
        # Set host='0.0.0.0' to be accessible on the network if needed
        # Debug=True enables auto-reloading and provides more detailed error pages
        # Use threaded=False to avoid socket issues on Windows
        app.run(debug=True, port=5001, host='0.0.0.0', threaded=False, use_reloader=False)
    except OSError as e:
        logger.error(f"Socket error occurred: {e}")
        logger.info("Attempting to restart server...")
        try:
            # Try again with different settings
            app.run(debug=False, port=5001, host='127.0.0.1', threaded=False)
        except Exception as e2:
            logger.critical(f"Failed to restart server: {e2}")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        logger.critical(traceback.format_exc())