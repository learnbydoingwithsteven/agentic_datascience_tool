import React from 'react';

/**
 * Component for displaying AI model analysis results
 * @param {string} analysis - The analysis text from the Ollama model
 * @param {boolean} isLoading - Loading state indicator
 * @param {string} dataset - Name of the analyzed dataset
 * @param {string} model - Name of the Ollama model used
 * @param {string} analysisType - Type of analysis performed
 */
function AnalysisPanel({ analysis, isLoading, dataset, model, analysisType }) {
  // Format the analysis type for display
  const formatAnalysisType = (type) => {
    switch(type) {
      case 'general': return 'General Insights';
      case 'correlation': return 'Correlation Analysis';
      case 'recommendations': return 'Visualization Recommendations';
      default: return type;
    }
  };

  return (
    <div className="analysis-panel">
      <h3>AI Analysis Results</h3>
      <div className="analysis-metadata">
        <p><strong>Dataset:</strong> {dataset}</p>
        <p><strong>Model:</strong> {model}</p>
        <p><strong>Analysis Type:</strong> {formatAnalysisType(analysisType)}</p>
      </div>
      
      <div className="analysis-content">
        {isLoading ? (
          <div className="loading-indicator">
            <p>Analyzing data with {model}...</p>
            <div className="spinner"></div>
          </div>
        ) : analysis ? (
          <div className="analysis-text">
            {/* Split analysis into paragraphs for better readability */}
            {analysis.split('\n').map((paragraph, index) => (
              <p key={index}>{paragraph}</p>
            ))}
          </div>
        ) : (
          <p>No analysis available. Click "Analyze Data" to generate insights.</p>
        )}
      </div>
    </div>
  );
}

export default AnalysisPanel;