import os
import pandas as pd
import requests
import json
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

DATASET_FOLDER = 'datasets'
OLLAMA_API_URL = 'http://localhost:11434/api'  # Default Ollama API URL

# --- API Endpoints ---

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """Lists available CSV datasets in the datasets folder."""
    try:
        # Check if datasets directory exists, if not create it
        if not os.path.exists(DATASET_FOLDER):
            os.makedirs(DATASET_FOLDER)
            # Create a sample dataset if none exists
            create_sample_dataset()
            
        datasets = [f.replace('.csv', '') for f in os.listdir(DATASET_FOLDER) if f.endswith('.csv')]
        return jsonify(datasets)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/<dataset_name>', methods=['GET'])
def get_dataset_data(dataset_name):
    """Returns the content of a specific dataset as JSON."""
    try:
        file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            return jsonify({"error": "Dataset not found"}), 404

        df = pd.read_csv(file_path)
        # Convert dataframe to JSON format suitable for charting libraries
        # 'records' format is often easy to work with [{col1: valA, col2: valB}, ...]
        data = df.to_dict(orient='records')
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """Lists available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            # Extract just the names for simplicity
            model_names = [model.get('name') for model in models]
            return jsonify(model_names)
        else:
            return jsonify({"error": f"Failed to fetch models: {response.text}"}), response.status_code
    except Exception as e:
        return jsonify({"error": f"Error connecting to Ollama: {str(e)}"}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Analyzes dataset using Ollama model and returns insights."""
    try:
        data = request.json
        dataset_name = data.get('dataset')
        model_name = data.get('model', 'llama3')
        analysis_type = data.get('analysis_type', 'general')
        
        # Get the dataset
        file_path = os.path.join(DATASET_FOLDER, f"{dataset_name}.csv")
        if not os.path.exists(file_path):
            return jsonify({"error": "Dataset not found"}), 404
            
        df = pd.read_csv(file_path)
        
        # Prepare data summary for the model
        data_summary = {
            "dataset_name": dataset_name,
            "columns": list(df.columns),
            "shape": df.shape,
            "sample": df.head(5).to_dict(orient='records'),
            "description": df.describe().to_dict()
        }
        
        # Prepare prompt based on analysis type
        if analysis_type == 'general':
            prompt = f"""Analyze this dataset and provide key insights:
            Dataset: {dataset_name}
            Columns: {data_summary['columns']}
            Shape: {data_summary['shape']}
            Sample data: {json.dumps(data_summary['sample'], indent=2)}
            Statistical summary: {json.dumps(data_summary['description'], indent=2)}
            
            Provide a concise analysis with the most important patterns and insights."""
        elif analysis_type == 'correlation':
            prompt = f"""Analyze correlations in this dataset:
            Dataset: {dataset_name}
            Columns: {data_summary['columns']}
            Sample data: {json.dumps(data_summary['sample'], indent=2)}
            
            Identify the strongest correlations between variables and explain their significance."""
        elif analysis_type == 'recommendations':
            prompt = f"""Based on this dataset, recommend visualization approaches:
            Dataset: {dataset_name}
            Columns: {data_summary['columns']}
            Shape: {data_summary['shape']}
            Sample data: {json.dumps(data_summary['sample'], indent=2)}
            
            Suggest the best visualization types for this data and explain why they would be effective."""
        else:
            prompt = f"""Provide a general analysis of this dataset:
            Dataset: {dataset_name}
            Columns: {data_summary['columns']}
            Sample data: {json.dumps(data_summary['sample'], indent=2)}"""
        
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "analysis": result.get('response', 'No analysis provided'),
                "model": model_name,
                "dataset": dataset_name
            })
        else:
            return jsonify({"error": f"Model inference failed: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_sample_dataset():
    """Creates a sample iris dataset if no datasets exist."""
    try:
        # Simple iris dataset
        iris_data = {
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
            'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa']
        }
        
        df = pd.DataFrame(iris_data)
        df.to_csv(os.path.join(DATASET_FOLDER, 'iris.csv'), index=False)
        
        # Simple sales dataset
        sales_data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'product': ['Widget A', 'Widget B', 'Widget A', 'Widget C', 'Widget B'],
            'quantity': [10, 15, 8, 12, 20],
            'price': [25.99, 15.49, 25.99, 32.50, 15.49],
            'revenue': [259.90, 232.35, 207.92, 390.00, 309.80]
        }
        
        df = pd.DataFrame(sales_data)
        df.to_csv(os.path.join(DATASET_FOLDER, 'sales.csv'), index=False)
        
        print("Created sample datasets: iris.csv and sales.csv")
    except Exception as e:
        print(f"Error creating sample datasets: {str(e)}")

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run on a port other than React's default (3000)