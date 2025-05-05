import React from 'react';

/**
 * Component for selecting a dataset from available options
 * @param {Array} datasets - List of available dataset names
 * @param {string} selectedDataset - Currently selected dataset
 * @param {function} onDatasetChange - Handler for dataset selection change
 * @param {boolean} isLoading - Loading state indicator
 */
function DatasetSelector({ datasets, selectedDataset, onDatasetChange, isLoading }) {
  // Handle selection change event
  const handleChange = (event) => {
    onDatasetChange(event.target.value);
  };

  return (
    <div className="selector-container">
      <h3>Dataset Selection</h3>
      <div className="selector-controls">
        <label htmlFor="dataset-select">Choose a dataset: </label>
        <select
          id="dataset-select"
          value={selectedDataset}
          onChange={handleChange}
          disabled={isLoading || datasets.length === 0}
          className="selector-dropdown"
        >
          <option value="">-- Select Dataset --</option>
          {datasets.map((ds) => (
            <option key={ds} value={ds}>
              {ds}
            </option>
          ))}
        </select>
        {isLoading && <span className="loading-indicator"> Loading datasets...</span>}
      </div>
    </div>
  );
}

export default DatasetSelector;