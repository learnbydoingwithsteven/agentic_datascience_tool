import React from 'react';
import Plot from 'react-plotly.js';

function PlotlyDashboard({ configs }) {
  // Expect configs to be an array of { data: [...], layout: {...} } objects or null/empty
  if (!configs || !Array.isArray(configs) || configs.length === 0) {
     // Don't render anything, or a placeholder, if no configs are available
     // console.log("No Plotly configs to display.");
    return null; // Or return <p>No Plotly plots generated or available.</p>;
  }

  // console.log("Rendering Plotly configs:", configs);

  return (
    <div>
      {configs.map((config, index) => {
        // Basic validation for each config object
        if (!config || typeof config !== 'object' || !config.data || !config.layout) {
          console.error(`Invalid Plotly config structure at index ${index}:`, config);
          return <p key={index} style={{color: 'red'}}>Invalid Plotly config at index {index}.</p>;
        }
        return (
          <div key={index} style={{ marginBottom: '20px', border: '1px solid #ddd', padding: '15px', borderRadius: '5px', background: '#fff' }}>
            {/* Add title from layout if available */}
            {config.layout?.title && <h4 style={{ margin: '0 0 10px 0' }}>{config.layout.title}</h4>}
            <Plot
              data={config.data} // Assumes backend provided correct structure with actual data
              layout={{
                ...config.layout,
                autosize: true,
                margin: { l: 50, r: 30, t: 30, b: 50 }
              }} // Assumes backend provided correct structure
              style={{ width: '100%', height: '350px' }} // Fixed height for side-by-side layout
              useResizeHandler={true} // Ensures plot resizes with container
              onError={(err) => console.error(`Plotly rendering error for plot ${index}:`, err)} // Add error handler
            />
            {/* Optionally display the config for debugging */}
            {/*
            <details style={{ marginTop: '10px' }}>
                <summary style={{ cursor: 'pointer' }}>Plotly Config {index + 1}</summary>
                <pre style={{ fontSize: '0.8em', background: '#eee', padding: '10px', borderRadius: '3px', maxHeight: '200px', overflowY: 'auto' }}>
                    {JSON.stringify(config, null, 2)}
                </pre>
            </details>
            */}
          </div>
        );
      })}
    </div>
  );
}

export default PlotlyDashboard;