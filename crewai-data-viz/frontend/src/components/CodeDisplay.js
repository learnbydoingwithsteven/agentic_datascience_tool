import React, { useState } from 'react';
import './CodeDisplay.css';

const CodeDisplay = ({ coderOutput, executorOutput }) => {
  const [activeTab, setActiveTab] = useState('coder');

  return (
    <div className="code-display">
      <div className="code-tabs">
        <button 
          className={`tab-button ${activeTab === 'coder' ? 'active' : ''}`}
          onClick={() => setActiveTab('coder')}
        >
          Python Code
        </button>
        <button 
          className={`tab-button ${activeTab === 'executor' ? 'active' : ''}`}
          onClick={() => setActiveTab('executor')}
        >
          Execution Results
        </button>
      </div>
      
      <div className="code-content">
        {activeTab === 'coder' && (
          <div className="code-panel">
            <h3>Python Visualization Code</h3>
            <pre className="code-block">
              <code>{coderOutput || 'No code generated yet.'}</code>
            </pre>
          </div>
        )}
        
        {activeTab === 'executor' && (
          <div className="code-panel">
            <h3>Code Execution Results</h3>
            <pre className="code-block">
              <code>{executorOutput || 'No execution results yet.'}</code>
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default CodeDisplay;
