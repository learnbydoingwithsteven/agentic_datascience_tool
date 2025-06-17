import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ModelSelector.css';

const API_BASE_URL = 'http://localhost:5001/api';

/**
 * Component for selecting an LLM model to use for visualization generation
 */
const ModelSelector = ({ selectedModel, onModelChange, isDisabled }) => {
  const [models, setModels] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await axios.get(`${API_BASE_URL}/models`);
        setModels(response.data || []);
        
        // If we have models and none is selected, select the first one
        if (response.data?.length > 0 && !selectedModel) {
          onModelChange(response.data[0].id);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        setError('Failed to load models');
        // Add default models as fallback
        const defaultModels = [
          { id: 'ollama/qwen3:4b', name: 'Qwen3 4B (Default)' },
          { id: 'ollama/llama3:8b', name: 'Llama3 8B' },
          { id: 'ollama/mistral:7b', name: 'Mistral 7B' }
        ];
        setModels(defaultModels);
        
        // If none is selected, select the default
        if (!selectedModel) {
          onModelChange(defaultModels[0].id);
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchModels();
  }, [onModelChange, selectedModel]);

  return (
    <div className="model-selector">
      <label htmlFor="model-select">LLM Model:</label>
      <select
        id="model-select"
        value={selectedModel || ''}
        onChange={(e) => onModelChange(e.target.value)}
        disabled={isDisabled || isLoading}
        className={isDisabled ? 'disabled' : ''}
      >
        <option value="" disabled>
          {isLoading ? 'Loading models...' : 'Select a model'}
        </option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default ModelSelector;
