import React from 'react';

/**
 * Component for selecting an Ollama model and analysis type
 * @param {Array} models - List of available Ollama models
 * @param {string} selectedModel - Currently selected model
 * @param {function} onModelChange - Handler for model selection change
 * @param {string} analysisType - Type of analysis to perform
 * @param {function} onAnalysisTypeChange - Handler for analysis type change
 * @param {function} onAnalyzeClick - Handler for analyze button click
 * @param {boolean} isLoading - Loading state indicator
 */
function ModelSelector({ 
  models, 
  selectedModel, 
  onModelChange, 
  analysisType, 
  onAnalysisTypeChange, 
  onAnalyzeClick, 
  isLoading 
}) {
  // Handle model selection change
  const handleModelChange = (event) => {
    onModelChange(event.target.value);
  };
  
  // Handle analysis type change
  const handleAnalysisTypeChange = (event) => {
    onAnalysisTypeChange(event.target.value);
  };

  return (
    <div className="selector-container">
      <h3>AI Model Analysis</h3>
      <div className="selector-controls">
        <label htmlFor="model-select">Ollama Model: </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={handleModelChange}
          disabled={isLoading || models.length === 0}
          className="selector-dropdown"
        >
          <option value="">-- Select Model --</option>
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
        
        <label htmlFor="analysis-type" className="ml-3">Analysis Type: </label>
        <select
          id="analysis-type"
          value={analysisType}
          onChange={handleAnalysisTypeChange}
          disabled={isLoading}
          className="selector-dropdown"
        >
          <option value="general">General Insights</option>
          <option value="correlation">Correlation Analysis</option>
          <option value="recommendations">Visualization Recommendations</option>
        </select>
        
        <button 
          onClick={onAnalyzeClick} 
          disabled={isLoading || !selectedModel} 
          className="analyze-button"
        >
          {isLoading ? 'Analyzing...' : 'Analyze Data'}
        </button>
      </div>
    </div>
  );
}

export default ModelSelector;