import React, { useState, useEffect } from 'react';
import ReactECharts from 'echarts-for-react';

/**
 * Component for displaying ECharts visualizations of the selected dataset
 * @param {Array} data - The dataset to visualize
 */
function EchartsDashboard({ data }) {
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
    return <p>No data available for ECharts visualizations.</p>;
  }

  // Determine numeric columns for plotting
  const numericColumns = columns.filter(col => {
    return typeof data[0][col] === 'number';
  });
  
  // Determine categorical columns for grouping
  const categoricalColumns = columns.filter(col => {
    return typeof data[0][col] === 'string';
  });
  
  // --- Bar Chart Configuration ---
  let barChartOption = {};
  
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
    
    barChartOption = {
      title: {
        text: `Bar Chart: Sum of ${valueCol} by ${categoryCol}`,
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: Object.keys(aggregatedData),
        name: categoryCol,
        nameLocation: 'middle',
        nameGap: 30
      },
      yAxis: {
        type: 'value',
        name: `Sum of ${valueCol}`,
        nameLocation: 'middle',
        nameGap: 30
      },
      series: [
        {
          name: valueCol,
          type: 'bar',
          data: Object.values(aggregatedData),
          itemStyle: {
            color: '#5470c6'
          }
        }
      ]
    };
  }
  
  // --- Pie Chart Configuration ---
  let pieChartOption = {};
  
  if (categoricalColumns.length > 0 && numericColumns.length > 0) {
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
    
    // Convert to series data format for pie chart
    const seriesData = Object.keys(aggregatedData).map(key => ({
      name: key,
      value: aggregatedData[key]
    }));
    
    pieChartOption = {
      title: {
        text: `Pie Chart: Distribution of ${valueCol} by ${categoryCol}`,
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c} ({d}%)'
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        data: Object.keys(aggregatedData)
      },
      series: [
        {
          name: valueCol,
          type: 'pie',
          radius: '50%',
          center: ['50%', '60%'],
          data: seriesData,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    };
  }
  
  // --- Line Chart Configuration ---
  let lineChartOption = {};
  
  if (numericColumns.length >= 2) {
    // Use the first two numeric columns
    const xCol = numericColumns[0];
    const yCol = numericColumns[1];
    
    // Sort data by x value for proper line chart
    const sortedData = [...data].sort((a, b) => a[xCol] - b[xCol]);
    
    lineChartOption = {
      title: {
        text: `Line Chart: ${yCol} vs ${xCol}`,
        left: 'center'
      },
      tooltip: {
        trigger: 'axis'
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'value',
        name: xCol,
        nameLocation: 'middle',
        nameGap: 30,
        data: sortedData.map(item => item[xCol])
      },
      yAxis: {
        type: 'value',
        name: yCol,
        nameLocation: 'middle',
        nameGap: 30
      },
      series: [
        {
          name: yCol,
          type: 'line',
          data: sortedData.map(item => item[yCol]),
          smooth: true,
          lineStyle: {
            width: 2,
            color: '#91cc75'
          },
          symbol: 'circle',
          symbolSize: 8
        }
      ]
    };
  }

  return (
    <div className="dashboard-container">
      <h2>ECharts Dashboard</h2>
      <div className="charts-grid">
        {/* Bar Chart */}
        <div className="chart-container">
          {numericColumns.length > 0 && categoricalColumns.length > 0 ? (
            <ReactECharts 
              option={barChartOption} 
              style={{ height: '400px', width: '100%' }} 
              className="echarts-chart"
            />
          ) : (
            <p>Insufficient data for bar chart (need at least 1 numeric and 1 categorical column)</p>
          )}
        </div>
        
        {/* Pie Chart */}
        <div className="chart-container">
          {numericColumns.length > 0 && categoricalColumns.length > 0 ? (
            <ReactECharts 
              option={pieChartOption} 
              style={{ height: '400px', width: '100%' }} 
              className="echarts-chart"
            />
          ) : (
            <p>Insufficient data for pie chart (need at least 1 numeric and 1 categorical column)</p>
          )}
        </div>
        
        {/* Line Chart */}
        <div className="chart-container">
          {numericColumns.length >= 2 ? (
            <ReactECharts 
              option={lineChartOption} 
              style={{ height: '400px', width: '100%' }} 
              className="echarts-chart"
            />
          ) : (
            <p>Insufficient numeric columns for line chart (need at least 2)</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default EchartsDashboard;