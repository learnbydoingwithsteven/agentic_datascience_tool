import React from 'react';
import ReactECharts from 'echarts-for-react';

function EchartsDashboard({ configs }) {
   // Expect configs to be an array of ECharts option objects or null/empty
  if (!configs || !Array.isArray(configs) || configs.length === 0) {
    // console.log("No ECharts configs to display.");
    return null; // Or return <p>No ECharts plots generated or available.</p>;
  }

  // console.log("Rendering ECharts configs:", configs);

  return (
    <div>
      {configs.map((config, index) => {
         // Basic validation for each config object
        if (!config || typeof config !== 'object' || !config.series) { // ECharts usually requires at least a 'series'
          console.error(`Invalid ECharts config structure at index ${index}:`, config);
          return <p key={index} style={{color: 'red'}}>Invalid ECharts config at index {index}.</p>;
        }
        return (
          <div key={index} style={{ marginBottom: '20px', border: '1px solid #ddd', padding: '15px', borderRadius: '5px', background: '#fff' }}>
             {/* Add a title if the config doesn't include one */}
             {!config.title?.text && <h4 style={{ margin: '0 0 10px 0' }}>EChart Plot {index + 1}</h4>}
             {config.title?.text && <h4 style={{ margin: '0 0 10px 0' }}>{config.title.text}</h4>}
             <ReactECharts
               option={{
                 ...config,
                 grid: { ...config.grid, containLabel: true, left: '3%', right: '4%', bottom: '8%' }
               }} // Assumes backend provided correct structure with actual data
               style={{ height: '350px', width: '100%' }}
               notMerge={true} // Important for dynamic updates
               lazyUpdate={true}
               onError={(err) => console.error(`ECharts rendering error for plot ${index}:`, err)} // Add error handler
               onEvents={{ // Optional: Add event handlers if needed
                  // 'click': (params) => console.log('ECharts click:', params),
               }}
             />
             {/* Optionally display the config for debugging */}
             {/*
             <details style={{ marginTop: '10px' }}>
                <summary style={{ cursor: 'pointer' }}>ECharts Config {index + 1}</summary>
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

export default EchartsDashboard;