import React, { useState, useEffect } from 'react';
import axios from 'axios';
// Using BrowserRouter for potential future routing, though not strictly needed for current layout
import { BrowserRouter as Router } from 'react-router-dom';
import DatasetSelector from './components/DatasetSelector';
import UserRequestInput from './components/UserRequestInput';
import ModelSelector from './components/ModelSelector';
import PlotlyDashboard from './components/PlotlyDashboard';
import EchartsDashboard from './components/EchartsDashboard';
import CodeDisplay from './components/CodeDisplay';
import './App.css';

// Configuration
const API_BASE_URL = 'http://localhost:5001/api';

// Example requests for different datasets to help users get started
const EXAMPLE_REQUESTS = {
  'iris': [
    'Show the distribution of sepal length by species',
    'Compare petal width vs petal length for each species',
    'What is the correlation between sepal width and sepal length?'
  ],
  'sales': [
    'Show revenue trends over time',
    'Compare sales by product category',
    'What products have the highest average price?'
  ],
  // Add more dataset-specific examples as needed
  'default': [
    'Show the distribution of values',
    'Find correlations between numerical columns',
    'Compare categories in the dataset'
  ]
};

function App() {
  // State management
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedModel, setSelectedModel] = useState('ollama/qwen3:4b');
  const [userRequest, setUserRequest] = useState('');
  const [plotlyConfigs, setPlotlyConfigs] = useState(null);
  const [echartsConfigs, setEchartsConfigs] = useState(null);
  const [coderOutput, setCoderOutput] = useState(null);
  const [executorOutput, setExecutorOutput] = useState(null);
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false);
  const [isGeneratingPlots, setIsGeneratingPlots] = useState(false);
  const [error, setError] = useState(null);
  const [crewResponse, setCrewResponse] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [isLoadingDatasetInfo, setIsLoadingDatasetInfo] = useState(false);

  // Get example requests for the selected dataset
  const getExampleRequests = () => {
    if (!selectedDataset) return [];
    return EXAMPLE_REQUESTS[selectedDataset] || EXAMPLE_REQUESTS['default'];
  };

  // Fetch dataset list on mount
  useEffect(() => {
    const fetchDatasets = async () => {
      setIsLoadingDatasets(true);
      setError(null);

      try {
        const response = await axios.get(`${API_BASE_URL}/datasets`);
        setDatasets(response.data || []);
      } catch (err) {
        console.error("Error fetching datasets:", err);
        let errorMsg = 'Failed to load datasets.';

        if (err.code === "ERR_NETWORK") {
          errorMsg += ' Is the backend server running?';
        } else if (err.response) {
          errorMsg += ` Server responded with ${err.response.status}: ${err.response.data?.error || err.message}`;
        } else {
          errorMsg += ` ${err.message}`;
        }

        setError(errorMsg);
        setDatasets([]);
      } finally {
        setIsLoadingDatasets(false);
      }
    };

    fetchDatasets();

    // Check API health
    axios.get(`${API_BASE_URL}/health`)
      .then(response => console.log("API Health:", response.data))
      .catch(err => console.warn("API Health check failed:", err));
  }, []);

  // Fetch dataset info when dataset changes
  useEffect(() => {
    if (!selectedDataset) {
      setDatasetInfo(null);
      return;
    }

    const fetchDatasetInfo = async () => {
      setIsLoadingDatasetInfo(true);

      try {
        const response = await axios.get(`${API_BASE_URL}/dataset_info/${selectedDataset}`);
        setDatasetInfo(response.data);
      } catch (err) {
        console.warn("Could not fetch dataset info:", err);
        // Don't show error to user, this is a non-critical feature
      } finally {
        setIsLoadingDatasetInfo(false);
      }
    };

    fetchDatasetInfo();
  }, [selectedDataset]);

  // Function to handle plot generation request
  const handleGeneratePlots = async (request) => {
    if (!selectedDataset) {
      setError("Please select a dataset first.");
      return;
    }

    setUserRequest(request);
    setIsGeneratingPlots(true);
    setError(null);
    setPlotlyConfigs(null);
    setEchartsConfigs(null);
    setCrewResponse(null);
    setCoderOutput(null);
    setExecutorOutput(null);

    try {
      console.log(`Sending request to generate plots: Dataset=${selectedDataset}, Request=${request}, Model=${selectedModel}`);

      const response = await axios.post(`${API_BASE_URL}/generate_plots`, {
        dataset_name: selectedDataset,
        user_request: request,
        model_id: selectedModel
      });

      console.log("Backend response:", response.data);

      // Process the response
      setPlotlyConfigs(response.data.plotlyConfigs || []);
      setEchartsConfigs(response.data.echartsConfigs || []);
      setCrewResponse(response.data.message || "Plots generated successfully.");
      setCoderOutput(response.data.coderOutput || "");
      setExecutorOutput(response.data.executorOutput || "");

      // Log raw output for debugging
      console.log("Crew Raw Output:", response.data.debug_raw_crew_output);
      console.log("Coder Output:", response.data.coderOutput);
      console.log("Executor Output:", response.data.executorOutput);
    } catch (err) {
      console.error("Error generating plots:", err);

      let errorMsg = 'Failed to generate plots.';
      let details = '';

      if (err.response) {
        // Use error message from backend if available
        errorMsg = err.response.data?.error || `Server error (${err.response.status})`;
        details = err.response.data?.details || err.message;
      } else if (err.request) {
        errorMsg = 'No response received from server. Is it running?';
      } else {
        errorMsg = err.message;
      }

      setError(`${errorMsg}${details ? ` (${details})` : ''}`);
    } finally {
      setIsGeneratingPlots(false);
    }
  };

  // Reset plots and request when dataset changes
  useEffect(() => {
    setPlotlyConfigs(null);
    setEchartsConfigs(null);
    setUserRequest('');
    setError(null);
    setCrewResponse(null);
    setCoderOutput(null);
    setExecutorOutput(null);
  }, [selectedDataset]);


  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>CrewAI Agentic Data Visualization</h1>
        </header>

        <main className="App-content">
          {/* Controls Section */}
          <section className="controls-section">
            <DatasetSelector
              datasets={datasets}
              selectedDataset={selectedDataset}
              onDatasetChange={setSelectedDataset}
              isLoading={isLoadingDatasets}
            />

            {/* Model Selector */}
            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
              isDisabled={isGeneratingPlots}
            />

            {/* Dataset Info Display */}
            {datasetInfo && !isLoadingDatasetInfo && (
              <div className="dataset-info-summary">
                <h3>Dataset: {datasetInfo.name}</h3>
                <p>{datasetInfo.rows} rows × {datasetInfo.columns} columns</p>
              </div>
            )}

            {/* User Request Input */}
            {selectedDataset && (
              <UserRequestInput
                onSubmit={handleGeneratePlots}
                isLoading={isGeneratingPlots}
                initialRequest={userRequest}
                exampleRequests={getExampleRequests()}
              />
            )}
          </section>

          {/* Status Messages */}
          <section className="status-section">
            {isGeneratingPlots && (
              <div className="status-loading">
                <p>Generating plots, please wait...</p>
                <div className="loading-spinner"></div>
              </div>
            )}
            {error && <div className="status-error">Error: {error}</div>}
            {crewResponse && !error && !isGeneratingPlots && (
              <div className="status-success">{crewResponse}</div>
            )}
          </section>

          {/* Visualization Display */}
          {(plotlyConfigs || echartsConfigs) && !isGeneratingPlots && (
            <section className="plots-display-section">
              <div className="request-context">
                <h2>Visualization Results</h2>
                <p>
                  Request: <strong>"{userRequest}"</strong> on dataset <strong>"{selectedDataset}"</strong>
                </p>
              </div>

              {/* Visualizations Container */}
              <div className="visualizations-container">
                {/* Plotly Visualizations */}
                {Array.isArray(plotlyConfigs) && plotlyConfigs.length > 0 && (
                  <div className="visualization-column">
                    <h3>Plotly Visualizations</h3>
                    <PlotlyDashboard configs={plotlyConfigs} />
                  </div>
                )}

                {/* ECharts Visualizations */}
                {Array.isArray(echartsConfigs) && echartsConfigs.length > 0 && (
                  <div className="visualization-column">
                    <h3>ECharts Visualizations</h3>
                    <EchartsDashboard configs={echartsConfigs} />
                  </div>
                )}
              </div>

              {/* No Visualizations Message */}
              {(!plotlyConfigs || plotlyConfigs.length === 0) &&
               (!echartsConfigs || echartsConfigs.length === 0) && (
                <div className="no-plots-message">
                  <p>The analysis did not result in any plot configurations for this request.</p>
                  <p>Try a different request or select another dataset.</p>
                </div>
              )}

              {/* Code Display */}
              {(coderOutput || executorOutput) && (
                <div>
                  <h3 style={{ marginTop: '30px', borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
                    Code & Execution Results
                  </h3>
                  <CodeDisplay
                    coderOutput={coderOutput}
                    executorOutput={executorOutput}
                  />
                </div>
              )}
            </section>
          )}
        </main>

        <footer className="App-footer">
          <p>Powered by CrewAI - Agentic Data Visualization</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;