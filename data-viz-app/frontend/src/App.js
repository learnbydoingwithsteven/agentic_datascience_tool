import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route, Link, useParams } from 'react-router-dom';
import DatasetSelector from './components/DatasetSelector';
import ModelSelector from './components/ModelSelector';
import PlotlyDashboard from './components/PlotlyDashboard';
import EchartsDashboard from './components/EchartsDashboard';
import AnalysisPanel from './components/AnalysisPanel';
import './App.css'; // You can add basic styling

// Make sure your backend is running (e.g., on port 5001)
const API_BASE_URL = 'http://localhost:5001/api';

function App() {
  // State management
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Ollama model integration
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [analysisType, setAnalysisType] = useState('general');
  const [analysis, setAnalysis] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);

  // Fetch list of datasets on component mount
  useEffect(() => {
    setLoading(true);
    axios.get(`${API_BASE_URL}/datasets`)
      .then(response => {
        setDatasets(response.data);
        setError(null);
      })
      .catch(err => {
        console.error("Error fetching datasets:", err);
        setError('Failed to load datasets.');
        setDatasets([]);
      })
      .finally(() => setLoading(false));
      
    // Fetch available Ollama models
    axios.get(`${API_BASE_URL}/models`)
      .then(response => {
        setModels(response.data);
        if (response.data.length > 0) {
          setSelectedModel(response.data[0]); // Select first model by default
        }
      })
      .catch(err => {
        console.error("Error fetching models:", err);
        setError('Failed to load Ollama models. Make sure Ollama is running.');
        setModels([]);
      });
  }, []);

  // Fetch data when selectedDataset changes
  useEffect(() => {
    if (!selectedDataset) {
      setData([]);
      return;
    }

    setLoading(true);
    setData([]); // Clear previous data
    setError(null);

    axios.get(`${API_BASE_URL}/data/${selectedDataset}`)
      .then(response => {
        setData(response.data);
      })
      .catch(err => {
        console.error(`Error fetching data for ${selectedDataset}:`, err);
        setError(`Failed to load data for ${selectedDataset}.`);
        setData([]);
      })
      .finally(() => setLoading(false));

  }, [selectedDataset]); // Dependency array
  
  // Function to request analysis from Ollama model
  const requestAnalysis = () => {
    if (!selectedDataset || !selectedModel) {
      setError('Please select both a dataset and a model for analysis');
      return;
    }
    
    setAnalysisLoading(true);
    setAnalysis(null);
    
    axios.post(`${API_BASE_URL}/analyze`, {
      dataset: selectedDataset,
      model: selectedModel,
      analysis_type: analysisType
    })
    .then(response => {
      setAnalysis(response.data.analysis);
      setError(null);
    })
    .catch(err => {
      console.error('Error analyzing data:', err);
      setError(`Analysis failed: ${err.response?.data?.error || err.message}`);
      setAnalysis(null);
    })
    .finally(() => setAnalysisLoading(false));
  };

  return (
    <Router>
      <div className="App">
        <h1>Data Visualization App with Ollama</h1>

        <div className="selectors-container">
          <DatasetSelector
            datasets={datasets}
            selectedDataset={selectedDataset}
            onDatasetChange={setSelectedDataset}
            isLoading={loading && datasets.length === 0}
          />
          
          <ModelSelector 
            models={models}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            analysisType={analysisType}
            onAnalysisTypeChange={setAnalysisType}
            onAnalyzeClick={requestAnalysis}
            isLoading={analysisLoading}
          />
        </div>

        {error && <p className="error-message">Error: {error}</p>}

        {selectedDataset && (
          <nav className="dashboard-nav">
            <Link to={`/plotly/${selectedDataset}`}>Plotly Dashboard</Link> |{' '}
            <Link to={`/echarts/${selectedDataset}`}>ECharts Dashboard</Link>
          </nav>
        )}

        {loading && data.length === 0 && selectedDataset && <p>Loading data...</p>}
        
        {/* Analysis Panel */}
        {(analysis || analysisLoading) && (
          <AnalysisPanel 
            analysis={analysis} 
            isLoading={analysisLoading} 
            dataset={selectedDataset}
            model={selectedModel}
            analysisType={analysisType}
          />
        )}

        <Routes>
          {/* Pass data down to dashboard components */}
          <Route path="/plotly/:datasetName" element={<PlotlyDashboard data={data} />} />
          <Route path="/echarts/:datasetName" element={<EchartsDashboard data={data} />} />
          <Route path="/" element={ <p>Please select a dataset to view dashboards.</p>} />
        </Routes>

      </div>
    </Router>
  );
}

export default App;