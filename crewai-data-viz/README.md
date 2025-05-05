# CrewAI Data Visualization System

This application provides an advanced agentic data visualization platform powered by CrewAI. It features a Flask backend that orchestrates AI agents to analyze datasets and generate visualizations, and a React frontend that displays interactive Plotly and ECharts visualizations side by side.

![CrewAI Data Visualization Tool](docs/screenshot.png)

*Screenshot: Side-by-side Plotly and ECharts visualizations with code display*

> Note: To add a screenshot, take a screenshot of the application in use and save it as `docs/screenshot.png`

## Current Status and Limitations

While the system has been enhanced with several improvements, it currently has some limitations in flexibility:

1. **Limited Dataset Adaptability**:
   - The system may struggle with certain dataset types and structures
   - Not all visualization requests can be handled automatically
   - Some dataset types may require manual intervention or specific request formatting

2. **Visualization Flexibility Constraints**:
   - The system is not as agile as originally designed
   - Complex visualization requests may result in errors or timeouts
   - Some chart types may not render correctly for all datasets

3. **Error Handling Improvements**:
   - Basic error handling has been implemented
   - The system now properly handles OS module references
   - Fallback mechanisms exist for common error cases

4. **Included Assets and Resources**:
   - The system includes various visualization libraries and assets
   - All included third-party assets are saved and properly referenced
   - Plotly, ECharts, and other visualization libraries are included

## Features

- **AI-Powered Analysis**: Uses CrewAI to orchestrate multiple AI agents that analyze data and generate visualizations
- **Enhanced Multi-Agent Collaboration**: Five specialized agents work together:
  - Data Analyst: Examines dataset structure and content
  - Plotly Strategist: Creates Plotly visualization configurations
  - ECharts Strategist: Creates ECharts visualization configurations
  - Visualization Coder: Writes Python code to generate visualizations
  - Visualization Executor: Executes the code and handles errors
- **Natural Language Requests**: Enter analysis requests in plain English
- **Side-by-Side Visualization Display**: Compare Plotly and ECharts visualizations in a clean, side-by-side layout
- **Code Transparency**: View and understand the Python code that generates the visualizations
- **Intelligent Error Handling**: Automatically fixes common issues like column name mismatches
- **CSV Dataset Support**: Works with any CSV dataset in the datasets folder

## Project Structure

```
crewai-data-viz/
├── backend/
│   ├── app.py            # Flask application with API endpoints
│   ├── crew_defs.py      # CrewAI agent and crew definitions
│   ├── data_tools.py     # Dataset loading and analysis tools with code execution capabilities
│   ├── datasets/         # Directory for CSV datasets
│   └── requirements.txt  # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── DatasetSelector.js    # Dataset selection component
    │   │   ├── UserRequestInput.js   # User request input component
    │   │   ├── PlotlyDashboard.js    # Plotly visualization dashboard
    │   │   ├── EchartsDashboard.js   # ECharts visualization dashboard
    │   │   └── CodeDisplay.js        # Code and execution results display component
    │   ├── App.js        # Main application component with side-by-side layout
    │   ├── index.js      # React entry point
    │   └── App.css       # Application styling with visualization layout
    └── package.json      # Node dependencies
```

## Prerequisites

- Python 3.8+ with pip
- Node.js 14+ and npm
- Ollama installed and running locally (or another LLM provider configured)

## Setup Instructions

### Backend Setup

1. Install Ollama from [ollama.ai](https://ollama.ai) and pull a model:

```bash
ollama pull llama3
```

2. Navigate to the backend directory and install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Place your CSV datasets in the `backend/datasets/` directory

4. Start the Flask server:

```bash
python app.py
```

The server will run on http://localhost:5001 by default.

### Frontend Setup

1. Navigate to the frontend directory and install dependencies:

```bash
cd frontend
npm install
```

2. Start the development server:

```bash
npm start
```

The application will open in your browser at http://localhost:3000.

## Usage

1. Select a dataset from the dropdown menu
2. Enter an analysis request in natural language (e.g., "Show the distribution of sepal length by species" or "Compare petal width vs petal length for each species")
3. Click "Generate Plots" to process your request
4. View the generated Plotly and ECharts visualizations side by side
5. Examine the Python code that generated the visualizations and its execution results in the Code Display section

## Configuration

- LLM settings can be configured in `backend/crew_defs.py`
- Default port settings can be modified in `backend/app.py` and `frontend/src/App.js`

## Extending the Application

### Adding New Datasets

Simply place CSV files in the `backend/datasets/` directory. They will automatically appear in the dataset selector.

### Using Different LLM Providers

Modify the LLM configuration in `backend/crew_defs.py` to use different providers like OpenAI, Anthropic, or others supported by LiteLLM.

### Customizing the Agent Architecture

The agent architecture can be customized in `backend/crew_defs.py`:
- Add new agents with specialized roles
- Modify the goals and tasks of existing agents
- Change the process flow between agents

### Enhancing the Code Execution Environment

The Python code execution environment can be extended in `backend/data_tools.py`:
- Add new libraries to the execution environment
- Implement additional error handling for specific use cases
- Create helper functions for common visualization tasks

## API Endpoints

The backend provides the following API endpoints:

- `/api/datasets`: List available datasets
- `/api/dataset_info/<dataset_name>`: Get information about a specific dataset
- `/api/generate_plots`: Generate visualizations based on a dataset and user request
- `/api/visualizations`: List all available visualizations
- `/api/visualizations/<filename>`: Get a specific visualization
- `/api/health`: Check if the API is running

## Troubleshooting

- **Backend Connection Issues**: Ensure the Flask server is running on port 5001
- **Missing Visualizations**: Check the browser console for errors and ensure your dataset has the columns mentioned in your request
- **LLM Errors**: Verify that Ollama is running and the specified model is available
- **Code Execution Errors**: Check the Execution Results tab in the Code Display section for error messages
- **Column Name Mismatches**: The system now automatically adapts to different column names and dataset structures
- **Timeout Issues**: The system has been improved to run without timeouts, allowing agents to complete their work
- **Server Restarts**: The Flask development server automatically restarts when code changes are detected, which is normal behavior

## Advanced Features

### Agent Architecture

The application uses a sophisticated multi-agent architecture powered by CrewAI:

1. **Data Analyst Agent**
   - Analyzes the dataset structure and content
   - Identifies relevant columns and relationships
   - Provides insights to guide visualization strategies

2. **Plotly Strategist Agent**
   - Creates Plotly visualization configurations
   - Selects appropriate chart types based on the data
   - Configures layout and styling for optimal data representation

3. **ECharts Strategist Agent**
   - Creates ECharts visualization configurations
   - Adapts the visualization strategy to ECharts' capabilities
   - Configures styling and interactive features

4. **Visualization Coder Agent**
   - Writes Python code to generate visualizations
   - Implements data processing and transformation
   - Creates effective visualizations using Plotly and other libraries

5. **Visualization Executor Agent**
   - Executes the Python code in a controlled environment
   - Handles errors and attempts to fix common issues
   - Converts visualization outputs to formats compatible with the frontend

These agents work sequentially, with each agent building on the work of the previous agents to create a comprehensive visualization solution.

### Side-by-Side Visualization

The application displays Plotly and ECharts visualizations side by side, allowing you to compare different visualization libraries' interpretations of the same data. This is particularly useful for:
- Comparing different chart types
- Evaluating which visualization library better represents your data
- Understanding the strengths of each library

### Code Transparency

The Code Display section shows:
- **Python Code**: The code generated by the Visualization Coder agent
- **Execution Results**: The output from running the code, including any errors

This transparency helps you:
- Understand how the visualizations are generated
- Learn data visualization techniques in Python
- Debug issues with the visualizations

## Future Enhancements

The CrewAI Data Visualization System is designed to be extensible. Here are some potential future enhancements:

1. **Additional Visualization Libraries**
   - Support for more visualization libraries like D3.js, Vega-Lite, or Matplotlib
   - Integration with dashboard frameworks like Dash or Streamlit

2. **Advanced Data Processing**
   - Time series analysis and forecasting
   - Statistical modeling and hypothesis testing
   - Machine learning integration for predictive analytics

3. **Enhanced User Interface**
   - Customization options for visualizations
   - Ability to save and share visualizations
   - Interactive data exploration features

4. **Collaborative Features**
   - Multi-user support with shared visualizations
   - Version control for visualizations and code
   - Commenting and annotation capabilities

5. **Deployment Options**
   - Containerization with Docker
   - Cloud deployment guides
   - Production-ready configuration options

6. **Visualization Storage and Management**
   - Persistent storage for generated visualizations
   - Visualization tagging and categorization
   - Search functionality for finding past visualizations

7. **Advanced Agent Capabilities**
   - Specialized agents for different types of data analysis
   - Agents that can learn from user feedback
   - Integration with external data sources and APIs

## Assets and Libraries

This project includes and uses the following third-party assets and libraries:

- **Plotly.js**: Interactive visualization library
- **ECharts**: Powerful charting and visualization library
- **React**: Frontend framework for building user interfaces
- **Flask**: Backend web framework
- **CrewAI**: Framework for orchestrating AI agents
- **Pandas**: Data manipulation and analysis library
- **NumPy**: Scientific computing library
- **LiteLLM**: Library for working with various LLM providers

### Saved Assets

All assets have been saved locally to ensure the application works reliably without external dependencies:

- **Frontend Libraries**: All JavaScript libraries are saved in the `frontend/node_modules` directory
- **Backend Libraries**: All Python libraries are installed in the virtual environment
- **Visualization Assets**: Plotly and ECharts assets are bundled with the frontend
- **Sample Datasets**: Sample datasets are included in the `backend/datasets` directory
- **Temporary Visualizations**: Generated visualizations are saved in the `backend/temp_visualizations` directory

These saved assets ensure that the application can run without requiring internet access after initial setup, making it more reliable for offline use and development.

## License

Apache License 2.0

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

The third-party libraries and assets used in this project are subject to their respective licenses:

- Plotly.js: MIT License
- ECharts: Apache License 2.0
- React: MIT License
- Flask: BSD License
- CrewAI: MIT License
- Pandas: BSD License
- NumPy: BSD License
- LiteLLM: MIT License