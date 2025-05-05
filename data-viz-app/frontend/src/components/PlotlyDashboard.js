import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

/**
 * Component for displaying Plotly visualizations of the selected dataset
 * @param {Array} data - The dataset to visualize
 */
function PlotlyDashboard({ data }) {
  // State to store column names for dynamic plot configuration
  const [columns, setColumns] = useState([]);
  
  // Extract column names from data when it changes
  useEffect(() => {
    if (data && data.length > 0) {
      setColumns(Object.keys(data[0]));
    }
  }, [data]);
  
  // Check if data is available
  if (!data || data.length === 0) {
    return <p>No data available for Plotly charts.</p>;
  }

  // Determine numeric columns for plotting
  const numericColumns = columns.filter(col => {
    return typeof data[0][col] === 'number';
  });
  
  // Determine categorical columns for grouping
  const categoricalColumns = columns.filter(col => {
    return typeof data[0][col] === 'string';
  });
  
  // --- Scatter Plot Configuration ---
  // Use the first two numeric columns for x and y if available
  const scatterData = [];
  
  if (numericColumns.length >= 2) {
    // If we have a categorical column, use it for color grouping
    if (categoricalColumns.length > 0) {
      // Group data by the first categorical column
      const groupedData = {};
      const categoryCol = categoricalColumns[0];
      
      data.forEach(item => {
        const category = item[categoryCol];
        if (!groupedData[category]) {
          groupedData[category] = { x: [], y: [] };
        }
        groupedData[category].x.push(item[numericColumns[0]]);
        groupedData[category].y.push(item[numericColumns[1]]);
      });
      
      // Create a trace for each category
      Object.keys(groupedData).forEach(category => {
        scatterData.push({
          x: groupedData[category].x,
          y: groupedData[category].y,
          mode: 'markers',
          type: 'scatter',
          name: category,
        });
      });
    } else {
      // Simple scatter plot without categories
      scatterData.push({
        x: data.map(item => item[numericColumns[0]]),
        y: data.map(item => item[numericColumns[1]]),
        mode: 'markers',
        type: 'scatter',
        marker: { color: 'blue' },
      });
    }
  }
  
  const scatterLayout = {
    title: `Scatter Plot: ${numericColumns[0]} vs ${numericColumns[1]}`,
    xaxis: { title: numericColumns[0] },
    yaxis: { title: numericColumns[1] },
    height: 400,
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
  };
  
  // --- Bar Chart Configuration ---
  const barData = [];
  
  if (numericColumns.length > 0 && categoricalColumns.length > 0) {
    // Use the first categorical column and first numeric column
    const categoryCol = categoricalColumns[0];
    const valueCol = numericColumns[0];
    
    // Aggregate data by category
    const aggregatedData = {};
    data.forEach(item => {
      const category = item[categoryCol];
      if (!aggregatedData[category]) {
        aggregatedData[category] = 0;
      }
      aggregatedData[category] += Number(item[valueCol]);
    });
    
    barData.push({
      x: Object.keys(aggregatedData),
      y: Object.values(aggregatedData),
      type: 'bar',
      marker: {
        color: 'rgb(158,202,225)',
        opacity: 0.8,
      }
    });
  }
  
  const barLayout = {
    title: numericColumns.length > 0 && categoricalColumns.length > 0 ?
      `Bar Chart: Sum of ${numericColumns[0]} by ${categoricalColumns[0]}` :
      'Bar Chart (Insufficient data for this chart type)',
    xaxis: { title: categoricalColumns[0] },
    yaxis: { title: `Sum of ${numericColumns[0]}` },
    height: 400,
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
  };
  
  // --- Histogram Configuration ---
  const histogramData = [];
  
  if (numericColumns.length > 0) {
    histogramData.push({
      x: data.map(item => item[numericColumns[0]]),
      type: 'histogram',
      opacity: 0.7,
      marker: {
        color: 'rgba(255, 100, 102, 0.7)',
      },
    });
  }
  
  const histogramLayout = {
    title: numericColumns.length > 0 ?
      `Histogram of ${numericColumns[0]}` :
      'Histogram (Insufficient data for this chart type)',
    xaxis: { title: numericColumns[0] },
    yaxis: { title: 'Count' },
    height: 400,
    margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
  };

  return (
    <div className="dashboard-container">
      <h2>Plotly Dashboard</h2>
      <div className="charts-grid">
        {/* Scatter Plot */}
        <div className="chart-container">
          {numericColumns.length >= 2 ? (
            <Plot
              data={scatterData}
              layout={scatterLayout}
              config={{ responsive: true }}
              className="plotly-chart"
            />
          ) : (
            <p>Insufficient numeric columns for scatter plot (need at least 2)</p>
          )}
        </div>
        
        {/* Bar Chart */}
        <div className="chart-container">
          {numericColumns.length > 0 && categoricalColumns.length > 0 ? (
            <Plot
              data={barData}
              layout={barLayout}
              config={{ responsive: true }}
              className="plotly-chart"
            />
          ) : (
            <p>Insufficient data for bar chart (need at least 1 numeric and 1 categorical column)</p>
          )}
        </div>
        
        {/* Histogram */}
        <div className="chart-container">
          {numericColumns.length > 0 ? (
            <Plot
              data={histogramData}
              layout={histogramLayout}
              config={{ responsive: true }}
              className="plotly-chart"
            />
          ) : (
            <p>Insufficient numeric columns for histogram (need at least 1)</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default PlotlyDashboard;