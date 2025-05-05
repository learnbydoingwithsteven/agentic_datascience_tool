import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Global styles
import App from './App'; // Main application component
import './App.css'; // App-specific styles

// Find the root element in your HTML (likely in public/index.html)
const rootElement = document.getElementById('root');

if (rootElement) {
  // Create a root for the React application
  const root = ReactDOM.createRoot(rootElement);

  // Render the App component within StrictMode for development checks
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} else {
  console.error("Failed to find the root element. Ensure your HTML has an element with ID 'root'.");
}