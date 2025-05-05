import React, { useState, useEffect } from 'react';
import './UserRequestInput.css'; // We'll create this CSS file later

/**
 * Component for entering and submitting data analysis requests
 * 
 * @param {function} onSubmit - Handler for submitting the request
 * @param {boolean} isLoading - Loading state indicator
 * @param {string} initialRequest - Optional initial request value
 * @param {array} exampleRequests - Optional array of example requests to show as suggestions
 */
function UserRequestInput({ onSubmit, isLoading, initialRequest = '', exampleRequests = [] }) {
  const [request, setRequest] = useState(initialRequest);
  const [error, setError] = useState('');
  const [showExamples, setShowExamples] = useState(false);

  // Update request if initialRequest prop changes
  useEffect(() => {
    if (initialRequest) {
      setRequest(initialRequest);
    }
  }, [initialRequest]);

  const handleSubmit = (event) => {
    event.preventDefault();
    const trimmedRequest = request.trim();
    
    // Validate request
    if (!trimmedRequest) {
      setError('Please enter an analysis request');
      return;
    }
    
    // Clear any previous error
    setError('');
    
    // Submit the request
    onSubmit(trimmedRequest);
  };

  const handleRequestChange = (e) => {
    setRequest(e.target.value);
    // Clear error when user starts typing
    if (error) setError('');
  };

  const selectExampleRequest = (example) => {
    setRequest(example);
    setShowExamples(false);
  };

  return (
    <div className="user-request-container">
      <form onSubmit={handleSubmit} className="user-request-form">
        <div className="input-group">
          <label htmlFor="user-request" className="request-label">
            Enter Analysis Request:
          </label>
          <div className="input-wrapper">
            <input
              type="text"
              id="user-request"
              value={request}
              onChange={handleRequestChange}
              placeholder="e.g., 'Show distribution of petal lengths'"
              className={`request-input ${error ? 'input-error' : ''}`}
              disabled={isLoading}
              aria-label="Enter your analysis request"
              aria-invalid={!!error}
              aria-describedby={error ? "request-error" : undefined}
            />
            {exampleRequests.length > 0 && (
              <button 
                type="button" 
                className="examples-toggle"
                onClick={() => setShowExamples(!showExamples)}
                aria-label="Show example requests"
              >
                Examples
              </button>
            )}
          </div>
          {error && <div id="request-error" className="error-message">{error}</div>}
        </div>
        
        <button 
          type="submit" 
          disabled={isLoading || !request.trim()} 
          className={`submit-button ${isLoading ? 'loading' : ''}`}
          aria-busy={isLoading}
        >
          {isLoading ? 'Generating...' : 'Generate Plots'}
        </button>
      </form>

      {showExamples && exampleRequests.length > 0 && (
        <div className="example-requests">
          <h4>Example Requests:</h4>
          <ul>
            {exampleRequests.map((example, index) => (
              <li key={index}>
                <button 
                  type="button" 
                  onClick={() => selectExampleRequest(example)}
                  className="example-button"
                >
                  {example}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default UserRequestInput;