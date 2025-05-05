# Data Visualization App with Ollama Integration

This application provides an interactive data visualization platform with AI-powered analysis using Ollama for local model inference. It features a Flask backend for data management and a React frontend with both Plotly and ECharts visualizations.

## Features

- **Dataset Management**: Upload, select, and visualize CSV datasets
- **Dual Visualization Frameworks**: Toggle between Plotly and ECharts dashboards
- **Dynamic Chart Generation**: Automatically creates appropriate visualizations based on dataset properties
- **AI-Powered Analysis**: Uses Ollama to analyze datasets and provide insights
- **Multiple Analysis Types**: General insights, correlation analysis, and visualization recommendations

## Project Structure

```
data-viz-app/
├── backend/
│   ├── app.py            # Flask application with Ollama integration
│   ├── datasets/         # Directory for CSV datasets
│   └── requirements.txt  # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── DatasetSelector.js    # Dataset selection component
    │   │   ├── ModelSelector.js      # Ollama model selection component
    │   │   ├── AnalysisPanel.js      # Display for AI analysis results
    │   │   ├── PlotlyDashboard.js    # Plotly visualization dashboard
    │   │   └── EchartsDashboard.js   # ECharts visualization dashboard
    │   ├── App.js        # Main application component
    │   ├── index.js      # React entry point
    │   └── App.css       # Application styling
    └── package.json      # Node dependencies
```

## Prerequisites

- Python 3.7+ with pip
- Node.js and npm
- Ollama installed and running locally

## Setup Instructions

### 1. Install and Run Ollama

First, make sure you have Ollama installed and running. You can download it from [ollama.ai](https://ollama.ai).

After installation, pull a model (e.g., llama3):

```bash
ollama pull llama3
```

Ensure Ollama is running and accessible at http://localhost:11434.

### 2. Backend Setup

```bash
# Navigate to the backend directory
cd data-viz-app/backend

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py
```

The backend will start on http://localhost:5001.

### 3. Frontend Setup

```bash
# Navigate to the frontend directory
cd data-viz-app/frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will start on http://localhost:3000.

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Select a dataset from the dropdown menu
3. Choose an Ollama model and analysis type
4. Click "Analyze Data" to generate AI insights
5. Toggle between Plotly and ECharts dashboards to view different visualizations

## Sample Datasets

The application includes two sample datasets:

- **iris.csv**: A classic dataset containing measurements of iris flowers
- **sales.csv**: A simple sales dataset with product and revenue information

You can add your own CSV files to the `backend/datasets/` directory.

## Extending the Application

### Adding New Chart Types

To add new chart types, modify the `PlotlyDashboard.js` or `EchartsDashboard.js` files to include additional chart configurations.

### Supporting More Analysis Types

To add new analysis types, update the `ModelSelector.js` component to include additional options and modify the `/api/analyze` endpoint in `app.py` to handle the new analysis types.

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running and accessible at http://localhost:11434
- **Missing Models**: Use `ollama list` to check available models and `ollama pull <model>` to download new ones
- **CORS Errors**: The backend has CORS enabled, but you may need to adjust settings if hosting on different domains

## License

MIT