# Agentic Data Science Visualization Tools

This repository contains two complementary data visualization applications that demonstrate different approaches to AI-powered data analysis and visualization. Both applications leverage local LLM inference through Ollama but implement different architectural approaches to generate insights and visualizations.

## Two Complementary Approaches to AI-Powered Data Visualization

### 1. CrewAI Data Visualization Tool (Multi-Agent Architecture)

The CrewAI-based tool implements an advanced **agentic architecture** where multiple specialized AI agents collaborate to analyze data and generate visualizations. This approach demonstrates the power of agent-based systems in data science workflows.

**Key Architectural Features:**
- **Multi-Agent System**: Five specialized agents work together in a coordinated workflow
- **Agent Specialization**: Each agent has a specific role and expertise (analysis, visualization strategy, coding)
- **Sequential Process Flow**: Agents build on each other's outputs in a defined sequence
- **Code Generation & Execution**: Automatically generates and executes Python visualization code
- **Side-by-Side Visualization**: Displays Plotly and ECharts visualizations together for comparison

**Use Case Strengths:**
- Complex data analysis requiring multiple specialized skills
- Generating both code and visualizations from natural language requests
- Educational scenarios where seeing the generated code is valuable
- Projects requiring multiple visualization libraries for comparison

### 2. Data Visualization App with Ollama (Single-LLM Architecture)

The Ollama-based tool implements a **streamlined architecture** with a single LLM that provides analysis and recommendations, paired with pre-built visualization components. This approach demonstrates a more direct integration of AI into traditional visualization workflows.

**Key Architectural Features:**
- **Single LLM Integration**: One LLM handles all analysis and recommendations
- **Separate Analysis Types**: Distinct modes for general insights, correlations, and visualization suggestions
- **Pre-Built Visualization Components**: Visualizations are generated through templated components
- **Model Selection**: Flexibility to choose different Ollama models for analysis
- **Tabbed Dashboard Interface**: Switch between Plotly and ECharts in separate tabs

**Use Case Strengths:**
- Rapid insights and visualizations with minimal complexity
- Flexibility to try different LLM models for analysis
- Projects with well-defined visualization needs
- Scenarios where analysis and visualization are distinct steps

## Architectural Comparison

| Feature | CrewAI Tool | Ollama Tool |
|---------|-------------|-------------|
| **Architecture** | Multi-agent system | Single LLM with visualization components |
| **AI Integration** | Deep integration with code generation | Analysis and recommendations only |
| **Visualization Generation** | Dynamic code generation | Pre-built visualization templates |
| **User Interface** | Side-by-side visualizations | Tabbed visualization dashboards |
| **Complexity** | Higher (multiple agents, code execution) | Lower (direct LLM integration) |
| **Flexibility** | More adaptable to complex requests | More structured analysis types |
| **Code Transparency** | Shows generated Python code | No code visibility |

## Features

### CrewAI Data Visualization Tool

- **Enhanced Multi-Agent Collaboration**: Five specialized agents work together:
  - Data Analyst: Examines dataset structure and content
  - Plotly Strategist: Creates Plotly visualization configurations
  - ECharts Strategist: Creates ECharts visualization configurations
  - Visualization Coder: Writes Python code to generate visualizations
  - Visualization Executor: Executes the code and handles errors
- **Natural Language Requests**: Enter analysis requests in plain English
- **Automatic Visualization Generation**: AI agents determine the most appropriate visualizations
- **Side-by-Side Visualization Display**: Compare Plotly and ECharts visualizations in a clean layout
- **Code Transparency**: View the Python code that generates the visualizations
- **Intelligent Error Handling**: Automatically fixes common issues like column name mismatches
- **Fallback Mechanisms**: Direct handlers for common visualization requests

### Data Visualization App with Ollama

- **Local LLM Integration**: Uses Ollama for on-device inference
- **Multiple Analysis Types**: General insights, correlation analysis, and visualization recommendations
- **Interactive Dashboards**: Separate dashboards for Plotly and ECharts visualizations
- **Model Selection**: Choose from available Ollama models
- **Dataset Management**: Upload, select, and visualize CSV datasets
- **Dynamic Chart Generation**: Automatically creates appropriate visualizations based on dataset properties

## Directory Structure

```
.
├── crewai-data-viz/       # CrewAI-based visualization tool
│   ├── backend/           # Flask backend with CrewAI integration
│   └── frontend/          # React frontend
│
├── data-viz-app/          # Ollama-based visualization tool
│   ├── backend/           # Flask backend with Ollama integration
│   └── frontend/          # React frontend
│
└── datasets/              # Shared datasets directory
```

## Setup and Installation

Each application has its own setup instructions. Please refer to the README files in each project directory:

- [CrewAI Data Visualization Tool README](./crewai-data-viz/README.md)
- [Data Visualization App with Ollama README](./data-viz-app/README.md)

## Prerequisites

- Python 3.8+ with pip
- Node.js 14+ and npm
- Ollama installed and running locally (for both applications)

## Quick Start

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai) and pull a model:

```bash
ollama pull llama3
```

### 2. Start the CrewAI Data Visualization Tool

```bash
# Terminal 1 - Backend
cd crewai-data-viz/backend
pip install -r requirements.txt
python app.py

# Terminal 2 - Frontend
cd crewai-data-viz/frontend
npm install
npm start
```

The application will be available at http://localhost:3000

### 3. Start the Data Visualization App with Ollama

```bash
# Terminal 3 - Backend
cd data-viz-app/backend
pip install -r requirements.txt
python app.py

# Terminal 4 - Frontend
cd data-viz-app/frontend
npm install
npm start
```

This application will be available at a different port (typically http://localhost:3001)

## Usage Comparison

### CrewAI Data Visualization Tool

1. **Select a dataset** from the dropdown menu (e.g., iris, sales)
2. **Enter a natural language request** in the text field, such as:
   - "Compare petal width vs petal length for each species"
   - "Show a box plot of all columns"
   - "Create a scatter plot of sepal length vs sepal width colored by species"
3. **Click "Generate Visualizations"** to start the agent collaboration process
4. **View the results**:
   - Plotly and ECharts visualizations appear side by side
   - The Python code that generated the visualizations is displayed below
   - Execution results show any output or errors from the code execution

**Example Requests for the Iris Dataset:**
- "Show the distribution of sepal length by species"
- "Compare petal width vs petal length for each species"
- "What is the correlation between sepal width and sepal length?"
- "Create a box plot showing the distribution of all measurements"

### Data Visualization App with Ollama

1. **Select a dataset** from the dropdown menu
2. **Choose an Ollama model** from the available options (e.g., llama3, mistral)
3. **Select an analysis type**:
   - **General Analysis**: Overall dataset insights and patterns
   - **Correlation Analysis**: Relationships between variables
   - **Visualization Recommendations**: Suggested charts for the dataset
4. **Click "Analyze Data"** to generate AI insights
5. **Review the analysis** in the panel below
6. **Navigate between visualization tabs**:
   - **Plotly Dashboard**: Interactive Plotly visualizations
   - **ECharts Dashboard**: Interactive ECharts visualizations

**Analysis Types Explained:**
- **General Analysis**: Provides an overview of the dataset, including key statistics, data types, and notable patterns
- **Correlation Analysis**: Focuses on relationships between variables, highlighting strong correlations and potential causations
- **Visualization Recommendations**: Suggests specific chart types that would be appropriate for the dataset and explains why

## Extending the Applications

### Adding New Datasets

Place CSV files in the respective `datasets` directories of each application.

### Using Different LLM Providers

- For the CrewAI tool, modify the LLM configuration in `crewai-data-viz/backend/crew_defs.py`
- For the Ollama app, modify the API configuration in `data-viz-app/backend/app.py`

## Recent Improvements to the CrewAI Tool

The CrewAI Data Visualization Tool has recently undergone significant improvements to enhance stability, performance, and visualization quality:

### Stability Enhancements

- **Socket Error Handling**: Improved handling of socket errors that were causing server crashes
- **Timeout Management**: Added timeout mechanisms to prevent hanging processes
- **Error Recovery**: Enhanced error handling with automatic recovery from common failure modes
- **Server Configuration**: Optimized Flask server settings for better stability on Windows

### Visualization Improvements

- **Direct Visualization Handlers**: Added specialized handlers for common visualization requests:
  - Petal width vs petal length scatter plots
  - Box plots of all columns
- **Improved JSON Parsing**: Enhanced extraction of visualization configurations from LLM outputs
- **Fallback Visualizations**: Created intelligent defaults when agent-generated visualizations fail
- **Side-by-Side Layout**: Refined the layout to better display Plotly and ECharts visualizations together

### LLM Integration Enhancements

- **Boxed JSON Handling**: Added support for the specific JSON format used by the Qwen model
- **Prompt Optimization**: Refined agent prompts for better visualization generation
- **Default Configurations**: Created dataset-specific default visualizations for common requests
- **Code Generation Improvements**: Enhanced Python code generation for more reliable execution

## Technical Implementation Comparison

| Aspect | CrewAI Tool | Ollama Tool |
|--------|-------------|-------------|
| **Backend Framework** | Flask | Flask |
| **Frontend Framework** | React | React |
| **LLM Integration** | Via CrewAI with LiteLLM | Direct Ollama API calls |
| **Agent Framework** | CrewAI | None (single LLM) |
| **Visualization Libraries** | Plotly, ECharts | Plotly, ECharts |
| **Code Generation** | Dynamic Python code | None |
| **Code Execution** | Sandboxed Python environment | None |
| **Error Handling** | Multi-level (agent, execution, fallback) | Basic API error handling |
| **Visualization Layout** | Side-by-side | Tabbed interface |
| **Dataset Handling** | CSV files in datasets directory | CSV files in datasets directory |

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
