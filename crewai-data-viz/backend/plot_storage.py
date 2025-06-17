import os
import json
import uuid
import shutil
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger('crewai-viz-plot-storage')

# Define constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualizations')
TEMP_VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'temp_visualizations')

# Ensure directories exist
for directory in [VISUALIZATIONS_DIR, TEMP_VISUALIZATIONS_DIR]:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory at {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

class PlotStorage:
    """Manages storage and retrieval of visualization plots.
    
    This class handles the persistent storage of visualization plots,
    including both temporary and permanent storage options.
    """
    
    @staticmethod
    def save_plot(plot_data: Dict[str, Any], plot_type: str, dataset_name: str) -> str:
        """Save a plot configuration to storage.
        
        Args:
            plot_data: The plot configuration data
            plot_type: Type of plot ('plotly' or 'echarts')
            dataset_name: Name of the dataset used for the plot
            
        Returns:
            The unique ID of the saved plot
        """
        try:
            # Generate a unique ID for the plot
            plot_id = str(uuid.uuid4())
            
            # Create metadata for the plot
            metadata = {
                'id': plot_id,
                'type': plot_type,
                'dataset': dataset_name,
                'created_at': datetime.now().isoformat(),
                'data': plot_data
            }
            
            # Save to both temporary and permanent storage
            temp_path = os.path.join(TEMP_VISUALIZATIONS_DIR, f'{plot_type}_{plot_id}.json')
            perm_path = os.path.join(VISUALIZATIONS_DIR, f'{plot_type}_{plot_id}.json')
            
            with open(temp_path, 'w') as f:
                json.dump(metadata, f)
            
            # Copy to permanent storage
            shutil.copy2(temp_path, perm_path)
            
            logger.info(f"Successfully saved {plot_type} plot {plot_id} for dataset {dataset_name}")
            return plot_id
            
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return None
    
    @staticmethod
    def get_plot(plot_id: str, plot_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve a plot configuration by ID.
        
        Args:
            plot_id: The unique ID of the plot
            plot_type: Type of plot ('plotly' or 'echarts')
            
        Returns:
            The plot configuration data if found, None otherwise
        """
        try:
            # Try permanent storage first
            perm_path = os.path.join(VISUALIZATIONS_DIR, f'{plot_type}_{plot_id}.json')
            if os.path.exists(perm_path):
                with open(perm_path, 'r') as f:
                    return json.load(f)
            
            # Fall back to temporary storage
            temp_path = os.path.join(TEMP_VISUALIZATIONS_DIR, f'{plot_type}_{plot_id}.json')
            if os.path.exists(temp_path):
                with open(temp_path, 'r') as f:
                    return json.load(f)
            
            logger.warning(f"Plot {plot_id} not found in storage")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving plot {plot_id}: {e}")
            return None
    
    @staticmethod
    def list_plots(dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available plots, optionally filtered by dataset.
        
        Args:
            dataset_name: Optional name of dataset to filter by
            
        Returns:
            List of plot metadata
        """
        plots = []
        try:
            # List plots from permanent storage
            for filename in os.listdir(VISUALIZATIONS_DIR):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(VISUALIZATIONS_DIR, filename), 'r') as f:
                            plot_data = json.load(f)
                            if dataset_name is None or plot_data.get('dataset') == dataset_name:
                                plots.append(plot_data)
                    except Exception as e:
                        logger.error(f"Error reading plot file {filename}: {e}")
                        continue
            
            return sorted(plots, key=lambda x: x.get('created_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing plots: {e}")
            return []
    
    @staticmethod
    def cleanup_temp_plots(max_age_hours: int = 24) -> None:
        """Clean up old temporary plot files.
        
        Args:
            max_age_hours: Maximum age of temporary files in hours
        """
        try:
            current_time = datetime.now()
            for filename in os.listdir(TEMP_VISUALIZATIONS_DIR):
                file_path = os.path.join(TEMP_VISUALIZATIONS_DIR, filename)
                file_age = datetime.fromtimestamp(os.path.getctime(file_path))
                age_hours = (current_time - file_age).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed old temporary plot file: {filename}")
                    except Exception as e:
                        logger.error(f"Error removing temporary file {filename}: {e}")
                        
        except Exception as e:
            logger.error(f"Error during temporary plot cleanup: {e}")