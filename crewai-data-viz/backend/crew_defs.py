import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
import pandas as pd
import json
import re
import io
import traceback
import litellm

# Import tools and helper
from data_tools import dataset_info_tool, load_dataframe, python_code_execution_tool

load_dotenv()

# --- Configure LLM ---
# Use the correct format for CrewAI with Ollama
from crewai import LLM

# Set litellm to verbose mode for debugging
litellm.set_verbose = True

# Function to create LLM instance with the specified model
def create_llm(model_id="ollama/qwen3:4b"):
    """Create an LLM instance with the specified model.

    Args:
        model_id: ID of the model to use, in the format "provider/model_name"
                 e.g., "ollama/qwen3:4b", "ollama/llama3:8b"

    Returns:
        LLM instance configured with the specified model
    """
    # Default to Qwen3 4B if model_id is invalid
    if not model_id or not isinstance(model_id, str):
        model_id = "ollama/qwen3:4b"

    # Extract provider and model name
    parts = model_id.split('/')
    if len(parts) != 2:
        print(f"Invalid model_id format: {model_id}, using default")
        model_id = "ollama/qwen3:4b"
        parts = ["ollama", "qwen3:4b"]

    provider, model_name = parts

    # Configure LLM based on provider
    if provider.lower() == "ollama":
        return LLM(model=model_id, base_url="http://localhost:11434")
    else:
        # For other providers, we would add configuration here
        print(f"Unsupported provider: {provider}, using Ollama")
        return LLM(model="ollama/qwen3:4b", base_url="http://localhost:11434")

# Create default LLM instance
llm = create_llm("ollama/qwen3:4b")

# Ensure LLM is configured
if llm is None:
    raise ValueError("LLM not configured. Please set up the llm variable.")

# --- Agent Goal Templates ---
# Store the original goal strings with placeholders
DATASET_ANALYST_GOAL_TEMPLATE = (
    'Analyze the structure and content of the dataset ({dataset_name}) to understand its columns, data types, and potential relationships relevant '
    'to the user request: "{user_request}". Focus on:\n'
    '1. Identifying patterns, relationships, and characteristics that would be useful for creating effective visualizations\n'
    '2. Determining which columns would be most relevant for the user\'s request\n'
    '3. Suggesting specific visualization types that would best represent the data\n'
    '4. Identifying any data preprocessing steps that might be necessary (date conversion, aggregation, etc.)\n'
    '5. Highlighting any interesting insights or anomalies in the data\n\n'
    'Your analysis will guide the visualization specialists in creating the most effective and insightful visualizations possible.'
)
PLOTLY_STRATEGIST_GOAL_TEMPLATE = (
    'Based on the data analysis and user request ("{user_request}"), devise 2-4 suitable **Plotly** chart configurations '
    'for exploratory analysis. For each configuration:\n'
    '1. Choose the most appropriate chart type based on the data and user request\n'
    '2. Include a \'data\' array with objects containing \'type\', \'x_column\', \'y_column\', and other relevant properties\n'
    '3. Include a \'layout\' object with title, axes, and other display settings\n'
    '4. Ensure the configuration is valid JSON and directly usable by react-plotly.js\n'
    '5. Add comments explaining your visualization choices\n\n'
    'Be creative and consider various chart types including: line charts, bar charts, scatter plots, box plots, histograms, '
    'heatmaps, pie charts, and bubble charts. Focus on creating visualizations that tell a clear story about the data.\n\n'
    'Output ONLY the configurations as a list of valid JSON strings compatible with `react-plotly.js`. '
    'Each JSON object in the list must contain "data" and "layout" keys. '
    'Inside the "data" list, use "x_column" and "y_column" keys to specify the column names from the dataset to be used for the axes. '
    'Do NOT include the actual data values, only the column names.'
)
ECHARTS_STRATEGIST_GOAL_TEMPLATE = (
    'Based on the data analysis and user request ("{user_request}"), devise 2-4 suitable **ECharts** chart configurations '
    'for exploratory analysis. For each configuration:\n'
    '1. Choose the most appropriate chart type based on the data and user request\n'
    '2. Include appropriate series, xAxis, yAxis, and other relevant properties\n'
    '3. Use \'data_column\' to specify which columns to use\n'
    '4. Ensure the configuration is valid JSON and directly usable by echarts-for-react\n'
    '5. Add comments explaining your visualization choices\n\n'
    'Be creative and consider various chart types including: line charts, bar charts, scatter plots, pie charts, radar charts, '
    'tree maps, and heatmaps. Focus on creating interactive visualizations that allow users to explore the data effectively.\n\n'
    'Output ONLY the configurations as a list of valid JSON strings compatible with `echarts-for-react`. '
    'Each JSON object in the list should be a valid ECharts option object (containing `series`, `xAxis`, `yAxis`, etc.). '
    'Inside the `series` list items, use a "data_column" key to specify the primary column name for the series data. '
    'Inside the `xAxis` object, if it\'s categorical, use a "data_column" key to specify the column name for axis labels. '
    'Do NOT include the actual data values, only the column names.'
)
VISUALIZATION_CODER_GOAL_TEMPLATE = (
    'Based on the data analysis and user request ("{user_request}"), write Python code that uses Pandas, Plotly, and '
    'other libraries to process the data and create effective visualizations. Your code should:\n'
    '1. Load the dataset (which will be provided as \'df\')\n'
    '2. Perform any necessary data preprocessing (date conversion, aggregation, filtering, etc.)\n'
    '3. Create multiple visualizations (at least 3) that highlight key insights\n'
    '4. Use appropriate color schemes, labels, and titles\n'
    '5. Include comments explaining your code and visualization choices\n'
    '6. Use the \'save_plotly_fig(fig)\' function to save any Plotly figures you create\n\n'
    'Be creative and consider various visualization approaches. Your code should be clean, efficient, and well-documented.\n\n'
    'Output ONLY the Python code without any additional text or explanations.'
)
VISUALIZATION_EXECUTOR_GOAL_TEMPLATE = (
    'Execute the Python code provided by the Visualization Coder to generate visualizations for the user request: '
    '"{user_request}". Your responsibilities:\n'
    '1. Execute the provided Python code with the dataset\n'
    '2. Identify and fix any errors in the code\n'
    '3. Enhance the visualizations for clarity and insight\n'
    '4. Ensure multiple visualization types are generated\n'
    '5. Provide a summary of the visualizations and their insights\n'
    '6. Return the execution results, including all generated visualizations\n\n'
    'If the code doesn\'t produce effective visualizations, modify it to better address the user\'s request. '
    'Your goal is to ensure the user receives high-quality, insightful visualizations.'
)

# --- Helper Function to Parse LLM Output ---
def extract_json_configs(text: str) -> list | None:
    """Extracts a list of JSON objects (like Plotly/ECharts configs) from LLM output text."""
    if not text or not isinstance(text, str):
        return None

    results = []

    # Print the raw text for debugging
    print(f"Extracting JSON from text of length {len(text)}")
    print(f"First 200 chars: {text[:200]}")
    print(f"Last 200 chars: {text[-200:] if len(text) > 200 else text}")

    # Check for special case with boxed JSON (from Qwen model)
    boxed_json_pattern = r"\$\$(.*?)\$\$"
    boxed_matches = re.findall(boxed_json_pattern, text, re.DOTALL)
    if boxed_matches:
        print("Found boxed JSON pattern, attempting to extract")
        for boxed_content in boxed_matches:
            # Look for JSON-like content inside the box
            json_content = re.search(r"\{(.*?)\}", boxed_content, re.DOTALL)
            if json_content:
                try:
                    # Reconstruct proper JSON
                    json_str = "{" + json_content.group(1) + "}"
                    # Clean up the string
                    cleaned_str = re.sub(r',\s*}', '}', json_str)
                    cleaned_str = re.sub(r',\s*]', ']', cleaned_str)

                    # Try to parse
                    parsed = json.loads(cleaned_str)
                    if isinstance(parsed, dict):
                        print(f"Successfully parsed boxed JSON with keys: {list(parsed.keys())}")

                        # For Plotly, convert to proper format if needed
                        # Check if the essential keys ('type', 'x', 'y') are present in the parsed dictionary
                        # These keys are required to attempt a specific Plotly configuration structure
                        if all(k in parsed for k in ["type", "x", "y"]):
                            try: # Attempt to construct the Plotly configuration
                                # Create the data part of the Plotly configuration
                                plotly_data = {
                                    "type": parsed["type"],  # Get the chart type directly from parsed data; raises KeyError if missing
                                    "x_column": parsed["x"], # Get the x-axis column name directly; raises KeyError if missing
                                    "y_column": parsed["y"], # Get the y-axis column name directly; raises KeyError if missing
                                }
                                # Conditionally add 'mode' to the data if it's present in the parsed data
                                if "mode" in parsed:
                                    plotly_data["mode"] = parsed["mode"] # Add mode if specified

                                # Construct the full Plotly configuration object
                                plotly_config = {
                                    "data": [plotly_data], # The data array for Plotly
                                    "layout": { # The layout object for Plotly
                                        "title": f"{parsed['y']} vs {parsed['x']}", # Generate title using y and x column names; raises KeyError if missing
                                        "xaxis": {"title": parsed["x"]}, # Set x-axis title; raises KeyError if missing
                                        "yaxis": {"title": parsed["y"]}  # Set y-axis title; raises KeyError if missing
                                    }
                                }
                                results.append(plotly_config) # Add the successfully constructed Plotly config to results
                                print(f"Successfully converted boxed JSON to Plotly config.") # Log success
                            except KeyError as e: # Handle cases where essential keys are missing
                                # Log a warning indicating a missing key during Plotly config construction
                                print(f"Warning: Missing key {str(e)} in boxed JSON for Plotly config construction. Appending original parsed data.")
                                results.append(parsed) # Append the original parsed dictionary if conversion fails
                        else:
                            # This block executes if not all required keys ('type', 'x', 'y') for the specific Plotly conversion are present
                            # Log that the boxed JSON doesn't meet the criteria for this specific Plotly conversion
                            print(f"Boxed JSON does not meet criteria for specific Plotly conversion (missing 'type', 'x', or 'y'). Appending as is.")
                            results.append(parsed) # Append the original parsed dictionary, as it might be valid for ECharts or another format
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse boxed JSON: {e}")

    try:
        # First, try to parse the entire string as a JSON list directly
        parsed = json.loads(text)
        if isinstance(parsed, list):
            if all(isinstance(item, dict) for item in parsed):
                print("Successfully parsed entire text as JSON list")
                return parsed
        elif isinstance(parsed, dict):
            print("Successfully parsed entire text as JSON dict")
            return [parsed]
    except json.JSONDecodeError:
        print("Direct JSON parsing failed, trying regex patterns")

        # If direct parsing fails, try different regex patterns
        # Pattern for JSON inside code blocks
        json_pattern1 = r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```"
        # Pattern for JSON without code blocks
        json_pattern2 = r"(\[[\s\S]*?\]|\{[\s\S]*?\})"

        # Try the first pattern (with code blocks)
        matches = re.findall(json_pattern1, text, re.DOTALL)
        if not matches:
            print("No matches with code block pattern, trying without code blocks")
            # If no matches, try the second pattern
            matches = re.findall(json_pattern2, text, re.DOTALL)

        for json_str in matches:
            if json_str and json_str.strip():
                try:
                    # Clean up the string - remove any trailing commas before closing brackets
                    cleaned_str = re.sub(r',\s*}', '}', json_str)
                    cleaned_str = re.sub(r',\s*]', ']', cleaned_str)

                    parsed = json.loads(cleaned_str)
                    if isinstance(parsed, list):
                        print(f"Successfully parsed JSON list: {len(parsed)} items")
                        results.extend(item for item in parsed if isinstance(item, dict))
                    elif isinstance(parsed, dict):
                        print(f"Successfully parsed JSON dict with keys: {list(parsed.keys())}")
                        results.append(parsed)
                except json.JSONDecodeError as e:
                    print(f"Warning: Regex found potential JSON but failed to parse: {json_str[:100]}... Error: {e}")

                    # Default Plotly config fallback removed to ensure agents generate all plots.
                    # If parsing fails here, it means the LLM output for this specific segment was not valid JSON.
                    # The overall function will continue to try other parsing methods or rely on the executor agent.

    # If no results were found, we'll let the Python code execution tool handle it
    # The ensure_plotly_output function will create appropriate visualizations based on the dataset
    if not results:
        print("No JSON configs found from agents. The Python code execution tool will generate appropriate visualizations.")

    # Remove duplicates
    unique_results = []
    seen = set()
    for item in results:
        item_str = json.dumps(item, sort_keys=True)
        if item_str not in seen:
            unique_results.append(item)
            seen.add(item_str)
    return unique_results if unique_results else None

# --- Function to Run Crew and Format Output ---
def generate_visualizations(dataset_name: str, user_request: str, model_id: str = "ollama/qwen3:4b") -> dict:
    """Instantiates and runs the CrewAI process with manual interpolation.

    Args:
        dataset_name: Name of the dataset to visualize
        user_request: Natural language request for visualization
        model_id: ID of the LLM model to use (default: "ollama/qwen3:4b")

    Returns:
        Dictionary containing visualization configurations and outputs
    """

    # 1. Manually format goals
    try:
        formatted_analyst_goal = DATASET_ANALYST_GOAL_TEMPLATE.format(
            dataset_name=dataset_name, user_request=user_request
        )
        formatted_plotly_goal = PLOTLY_STRATEGIST_GOAL_TEMPLATE.format(
            user_request=user_request
        )
        formatted_echarts_goal = ECHARTS_STRATEGIST_GOAL_TEMPLATE.format(
            user_request=user_request
        )
        formatted_coder_goal = VISUALIZATION_CODER_GOAL_TEMPLATE.format(
            user_request=user_request
        )
        formatted_executor_goal = VISUALIZATION_EXECUTOR_GOAL_TEMPLATE.format(
            user_request=user_request
        )
    except KeyError as e:
         print(f"Error formatting goal templates: Missing key {e}")
         return {"error": f"Internal error formatting agent goals: Missing key {e}"}

    # Create LLM instance with the specified model
    model_llm = create_llm(model_id)

    # Log the model being used
    print(f"Using LLM model: {model_id}")

    # 2. Instantiate Agents with formatted goals
    dataset_analyst = Agent(
        role='Data Analyst',
        goal=formatted_analyst_goal,
        backstory='An expert data analyst skilled in Pandas and exploratory data analysis (EDA). You excel at identifying key characteristics of datasets, understanding relationships between variables, and extracting meaningful insights from data. You are thorough in your analysis and can interpret both explicit and vague user requests to determine the most relevant aspects of the data to focus on.',
        verbose=True, llm=model_llm, tools=[dataset_info_tool], allow_delegation=False
    )
    plotly_strategist = Agent(
        role='Plotly Visualization Strategist',
        goal=formatted_plotly_goal,
        backstory='A specialist in data visualization using Plotly with years of experience creating effective visualizations for diverse datasets. You can translate any data insights and user requests into appropriate Plotly JSON configurations (`data` and `layout` objects). You are creative and can suggest multiple visualization approaches for the same data. You prioritize clarity, relevance, and visual appeal. You output ONLY the JSON list.',
        verbose=True, llm=model_llm, allow_delegation=False
    )
    echarts_strategist = Agent(
        role='ECharts Visualization Strategist',
        goal=formatted_echarts_goal,
        backstory='A specialist in data visualization using Apache ECharts with extensive experience in creating interactive and visually appealing charts. You excel at creating ECharts option objects that effectively represent data insights according to user needs. You are skilled at selecting the most appropriate chart types for different data patterns and user requests. You output ONLY the JSON list.',
        verbose=True, llm=model_llm, allow_delegation=False
    )
    visualization_coder = Agent(
        role='Visualization Coder',
        goal=formatted_coder_goal,
        backstory='An expert Python programmer specializing in data visualization with mastery of Pandas, Plotly, Matplotlib, Seaborn, and other visualization libraries. You write clean, efficient, and robust code that handles edge cases gracefully. You are skilled at data preprocessing and can create sophisticated visualizations that address both simple and complex user requests. You can interpret vague requests and implement appropriate visualizations.',
        verbose=True, llm=model_llm, allow_delegation=False
    )
    visualization_executor = Agent(
        role='Visualization Executor',
        goal=formatted_executor_goal,
        backstory='A technical expert who executes Python code to generate visualizations and ensures they are properly formatted for the frontend. You have deep knowledge of data visualization principles and can debug and optimize visualization code. You are resourceful in handling errors and can adapt code to work with different datasets and requirements. You bridge the gap between code and visual output, ensuring high-quality results.',
        verbose=True, llm=model_llm, tools=[python_code_execution_tool], allow_delegation=False
    )

    # 3. Instantiate Tasks (using placeholder-free descriptions/expected_outputs)
    # Ensure these strings do NOT contain {dataset_name} or {user_request}
    analyze_task = Task(
        description="""
1. Use the Dataset Information Tool to thoroughly examine the specified dataset.
2. Analyze the dataset structure, including:
   - Column names, data types, and basic statistics
   - Distributions of numerical variables
   - Frequency counts of categorical variables
   - Correlations between numerical variables
   - Temporal patterns if date/time columns exist
3. Identify key insights relevant to the user request, including:
   - Which columns are most relevant to the request
   - Potential relationships that should be visualized
   - Any data quality issues that need to be addressed
4. For vague requests like 'show some plots', identify the most interesting aspects of the data.
5. Suggest specific visualization types that would be most effective.
6. Identify any necessary data preprocessing steps (date conversion, aggregation, etc.).
7. Provide a comprehensive analysis that will enable the visualization specialists to create effective visualizations.
        """,
        expected_output="""
A detailed textual summary of the dataset analysis, including:
1. Overview of the dataset structure and content
2. Key insights relevant to the user request
3. Specific columns that should be visualized
4. Recommended visualization types
5. Necessary data preprocessing steps
6. Any interesting patterns or anomalies discovered
        """,
        agent=dataset_analyst
    )
    plotly_task = Task(
        description="""
1. Review the data analysis summary and the original user request provided in the goal.
2. Identify 2-4 appropriate Plotly charts that would best represent the data and address the user's request.
3. For each chart:
   - Choose the most appropriate chart type based on the data and request
   - Determine which columns should be used for each axis or dimension
   - Design an effective layout with appropriate titles, labels, and styling
   - Consider color schemes that enhance data understanding
4. Consider a variety of chart types including:
   - Line charts for trends over time
   - Bar charts for comparing categories
   - Scatter plots for relationships between variables
   - Box plots for distributions and outliers
   - Heatmaps for correlation matrices
   - Pie charts for part-to-whole relationships
   - Histograms for distributions
5. Construct the Plotly JSON configuration list as specified in your goal.
6. Ensure each configuration is valid JSON and directly usable by react-plotly.js.
7. Output ONLY the list of JSON objects.
        """,
        expected_output="""
A list containing 2-4 valid Plotly JSON configuration objects, each having:
1. A "data" array with objects containing "type", "x_column", "y_column", and other relevant properties
2. A "layout" object with title, axes, and other display settings
3. Appropriate configuration for the specific chart type
Output ONLY the list, nothing else.
        """,
        agent=plotly_strategist,
        context=[analyze_task]
    )
    echarts_task = Task(
        description="""
1. Review the data analysis summary and the original user request provided in the goal.
2. Identify 2-4 appropriate ECharts visualizations that would best represent the data and address the user's request.
3. For each chart:
   - Choose the most appropriate chart type based on the data and request
   - Determine which columns should be used for each axis or dimension
   - Design an effective layout with appropriate titles, labels, and styling
   - Consider interactive features that enhance data exploration
4. Consider a variety of chart types including:
   - Line charts for trends over time
   - Bar charts for comparing categories
   - Scatter plots for relationships between variables
   - Pie charts for part-to-whole relationships
   - Tree maps for hierarchical data
   - Radar charts for multivariate data
   - Heatmaps for correlation matrices
5. Construct the ECharts JSON option object list as specified in your goal.
6. Ensure each configuration is valid JSON and directly usable by echarts-for-react.
7. Output ONLY the list of JSON objects.
        """,
        expected_output="""
A list containing 2-4 valid ECharts JSON option objects, each having:
1. Appropriate series, xAxis, yAxis, and other relevant properties
2. "data_column" keys to specify which columns to use
3. Proper configuration for the specific chart type
Output ONLY the list, nothing else.
        """,
        agent=echarts_strategist,
        context=[analyze_task]
    )
    coder_task = Task(
        description="""
1. Review the data analysis summary and the original user request provided in the goal.
2. Write Python code that uses Pandas, Plotly, and other libraries to process the data and create effective visualizations.
3. Your code should:
   - Handle the dataset (which will be provided as 'df')
   - Perform any necessary data preprocessing (date conversion, aggregation, filtering, etc.)
   - Create multiple visualizations (at least 3) that address the user's request
   - Use appropriate color schemes, labels, and titles
   - Include comments explaining your code and visualization choices
4. Include data preprocessing steps as needed:
   - Converting date/time columns to proper datetime format
   - Handling missing values appropriately
   - Creating derived features if needed
   - Aggregating data for summary visualizations
5. Create diverse visualization types that highlight different aspects of the data.
6. Use the 'save_plotly_fig(fig)' function to save any Plotly figures you create.
7. Output ONLY the Python code without any additional text or explanations.
        """,
        expected_output="""
Python code that:
1. Processes the data appropriately
2. Creates multiple diverse visualizations
3. Is well-commented and easy to understand
4. Uses appropriate visualization techniques for the data and request
5. Handles potential errors or edge cases
        """,
        agent=visualization_coder,
        context=[analyze_task]
    )
    executor_task = Task(
        description="""
1. Review the Python code provided by the Visualization Coder.
2. Use the Python Code Execution Tool to execute the code with the specified dataset.
3. The tool will automatically handle loading the dataset and executing the code.
4. Review the execution results and ensure the visualizations are properly generated.
5. If there are any errors:
   - Identify the cause of the error
   - Fix the code appropriately
   - Re-execute the code
6. Enhance the visualizations if needed:
   - Improve color schemes for better readability
   - Add or improve titles, labels, and legends
   - Adjust layouts for better presentation
7. If the code doesn't produce diverse visualizations, modify it to include additional visualization types.
8. Ensure the visualizations effectively address the user's request.
9. Output the execution results, including all generated visualizations.
        """,
        expected_output="""
Execution results including:
1. Status of code execution (success or errors encountered and fixed)
2. Multiple generated visualizations in formats compatible with the frontend
3. Brief description of each visualization and its insights
4. Any modifications made to the original code
        """,
        agent=visualization_executor,
        context=[analyze_task, coder_task]
    )

    # 4. Instantiate Crew
    viz_crew = Crew(
        agents=[dataset_analyst, plotly_strategist, echarts_strategist, visualization_coder, visualization_executor],
        tasks=[analyze_task, plotly_task, echarts_task, coder_task, executor_task],
        process=Process.sequential,
        verbose=True
    )

    # 5. Kickoff Crew (without inputs dictionary)
    crew_result = None
    try:
        # Run the crew directly without a timeout
        print("Starting CrewAI execution without timeout...")

        # Pass an empty inputs dictionary to avoid any string interpolation issues
        crew_result = viz_crew.kickoff(inputs={})
        print("CrewAI execution completed successfully")

    except Exception as e:
        print(f"Error during CrewAI kickoff: {e}")
        traceback.print_exc()
        return {
            "error": f"An error occurred during the CrewAI process: {e}",
            "plotly_configs": None, "echarts_configs": None,
            "raw_output": traceback.format_exc()
        }

    # 6. Process results
    raw_plotly_output = getattr(plotly_task.output, 'raw_output', None) or getattr(plotly_task.output, 'result', None)
    raw_echarts_output = getattr(echarts_task.output, 'raw_output', None) or getattr(echarts_task.output, 'result', None)
    raw_coder_output = getattr(coder_task.output, 'raw_output', None) or getattr(coder_task.output, 'result', None)
    raw_executor_output = getattr(executor_task.output, 'raw_output', None) or getattr(executor_task.output, 'result', None)

    if raw_plotly_output is None: print("Warning: Could not retrieve raw output from Plotly task."); raw_plotly_output = ""
    if raw_echarts_output is None: print("Warning: Could not retrieve raw output from ECharts task."); raw_echarts_output = ""
    if raw_coder_output is None: print("Warning: Could not retrieve raw output from Coder task."); raw_coder_output = ""
    if raw_executor_output is None: print("Warning: Could not retrieve raw output from Executor task."); raw_executor_output = ""

    print("\n--- Raw Plotly Output ---"); print(raw_plotly_output)
    print("\n--- Raw ECharts Output ---"); print(raw_echarts_output)
    print("\n--- Raw Coder Output ---"); print(raw_coder_output)
    print("\n--- Raw Executor Output ---"); print(raw_executor_output)

    # Process the outputs from the strategist agents
    plotly_configs = extract_json_configs(raw_plotly_output)
    echarts_configs = extract_json_configs(raw_echarts_output)

    # Process the output from the executor agent
    executor_results = None
    try:
        # Try to parse the executor output as JSON
        if raw_executor_output and isinstance(raw_executor_output, str):
            # Look for JSON in the executor output
            json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```|(\{[\s\S]*?\})"
            matches = re.findall(json_pattern, raw_executor_output, re.DOTALL)
            for match_group in matches:
                json_str = next((g for g in match_group if g and g.strip()), None)
                if json_str:
                    try:
                        executor_results = json.loads(json_str)
                        print(f"Successfully parsed executor results: {list(executor_results.keys())}")
                        break
                    except json.JSONDecodeError:
                        pass

            # If no JSON found in the output, try to parse the entire output
            if not executor_results:
                try:
                    executor_results = json.loads(raw_executor_output)
                    print(f"Successfully parsed executor results from raw output: {list(executor_results.keys())}")
                except json.JSONDecodeError:
                    print("Could not parse executor output as JSON")
    except Exception as e:
        print(f"Error processing executor output: {e}")

    # Extract Plotly and ECharts configs from executor results if available
    if executor_results and isinstance(executor_results, dict):
        if 'plotly_configs' in executor_results and executor_results['plotly_configs']:
            executor_plotly_configs = executor_results['plotly_configs']
            if isinstance(executor_plotly_configs, list) and all(isinstance(item, dict) for item in executor_plotly_configs):
                print(f"Found {len(executor_plotly_configs)} Plotly configs in executor results")
                # Add executor Plotly configs to the existing configs
                plotly_configs = (plotly_configs or []) + executor_plotly_configs

        if 'echarts_configs' in executor_results and executor_results['echarts_configs']:
            executor_echarts_configs = executor_results['echarts_configs']
            if isinstance(executor_echarts_configs, list) and all(isinstance(item, dict) for item in executor_echarts_configs):
                print(f"Found {len(executor_echarts_configs)} ECharts configs in executor results")
                # Add executor ECharts configs to the existing configs
                echarts_configs = (echarts_configs or []) + executor_echarts_configs

    # If no Plotly configs were extracted, we'll let the Python code execution tool handle it
    if not plotly_configs:
        print("No valid Plotly configs found from agents. The Python code execution tool will generate appropriate visualizations.")

    print("\n--- Parsed Plotly Configs ---"); print(json.dumps(plotly_configs, indent=2))
    print("\n--- Parsed ECharts Configs ---"); print(json.dumps(echarts_configs, indent=2))

    # --- Load actual data and embed it into the configs ---
    df = load_dataframe(dataset_name)
    final_plotly_configs = []
    final_echarts_configs = []

    if df is not None:
        # Plotly data injection
        if plotly_configs:
            for config in plotly_configs:
                try:
                    if not isinstance(config, dict) or 'data' not in config or not isinstance(config['data'], list): continue
                    processed_traces = []
                    valid_config = True
                    for trace in config['data']:
                        if not isinstance(trace, dict): valid_config = False; break
                        processed_trace = trace.copy()
                        if 'x_column' in processed_trace:
                            col_name = processed_trace.pop('x_column')
                            if col_name in df.columns: processed_trace['x'] = df[col_name].tolist()
                            else: valid_config = False; print(f"Error: Plotly x_column '{col_name}' not found."); break
                        if 'y_column' in processed_trace:
                            col_name = processed_trace.pop('y_column')
                            if col_name in df.columns: processed_trace['y'] = df[col_name].tolist()
                            else: valid_config = False; print(f"Error: Plotly y_column '{col_name}' not found."); break
                        processed_traces.append(processed_trace)
                    if valid_config:
                        final_config = config.copy(); final_config['data'] = processed_traces
                        final_plotly_configs.append(final_config)
                except Exception as e: print(f"Error processing Plotly config: {e}\nConfig: {config}"); traceback.print_exc()

        # ECharts data injection
        if echarts_configs:
             for config in echarts_configs:
                try:
                    if not isinstance(config, dict): continue
                    processed_config = config.copy(); valid_config = True
                    if 'series' in processed_config and isinstance(processed_config['series'], list):
                        processed_series_list = []
                        for series_item in processed_config['series']:
                             if not isinstance(series_item, dict): valid_config = False; break
                             processed_series = series_item.copy()
                             if 'data_column' in processed_series:
                                 col_name = processed_series.pop('data_column')
                                 if col_name in df.columns: processed_series['data'] = df[col_name].tolist()
                                 else: valid_config = False; print(f"Error: ECharts series data_column '{col_name}' not found."); break
                             processed_series_list.append(processed_series)
                        if not valid_config: continue
                        processed_config['series'] = processed_series_list

                    if 'xAxis' in processed_config:
                        axes = processed_config['xAxis'] if isinstance(processed_config['xAxis'], list) else [processed_config['xAxis']]
                        processed_axes = []
                        for axis in axes:
                            if isinstance(axis, dict) and 'data_column' in axis:
                                processed_axis = axis.copy(); col_name = processed_axis.pop('data_column')
                                if col_name in df.columns: processed_axis['data'] = df[col_name].unique().tolist()
                                else: valid_config = False; print(f"Error: ECharts xAxis data_column '{col_name}' not found."); break
                                processed_axes.append(processed_axis)
                            else: processed_axes.append(axis)
                        if not valid_config: continue
                        processed_config['xAxis'] = processed_axes[0] if not isinstance(processed_config['xAxis'], list) else processed_axes

                    if valid_config: final_echarts_configs.append(processed_config)
                except Exception as e: print(f"Error processing ECharts config: {e}\nConfig: {config}"); traceback.print_exc()
    else:
         return {
             "error": f"Failed to load dataset '{dataset_name}'.",
             "plotly_configs": None, "echarts_configs": None,
             "raw_output": str(crew_result) if crew_result else "DataFrame load failed"
         }

    return {
        "plotly_configs": final_plotly_configs if final_plotly_configs else [],
        "echarts_configs": final_echarts_configs if final_echarts_configs else [],
        "raw_output": str(crew_result),
        "coder_output": raw_coder_output,
        "executor_output": raw_executor_output
    }