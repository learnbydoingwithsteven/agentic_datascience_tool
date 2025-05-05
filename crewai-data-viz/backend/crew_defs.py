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

# Create LLM instance with the correct format
llm = LLM(model="ollama/qwen3:4b", base_url="http://localhost:11434")

# Ensure LLM is configured
if llm is None:
    raise ValueError("LLM not configured. Please set up the llm variable.")

# --- Agent Goal Templates ---
# Store the original goal strings with placeholders
DATASET_ANALYST_GOAL_TEMPLATE = (
    'Analyze the structure and content of the dataset ({dataset_name}) '
    'to understand its columns, data types, and potential relationships relevant '
    'to the user request: "{user_request}".'
)
PLOTLY_STRATEGIST_GOAL_TEMPLATE = (
    'Based on the data analysis and user request ("{user_request}"), devise 1-3 suitable **Plotly** chart configurations '
    'for exploratory analysis. Focus on common EDA plots like histograms (for distributions), '
    'scatter plots (for relationships between numerical variables), bar charts (for categorical counts/aggregations), '
    'and box plots (for distribution comparison across categories). '
    'Output ONLY the configurations as a list of valid JSON strings compatible with `react-plotly.js`. '
    'Each JSON object in the list must contain "data" and "layout" keys. '
    'Inside the "data" list, use "x_column" and "y_column" keys to specify the column names from the dataset to be used for the axes. '
    'Do NOT include the actual data values, only the column names.'
)
ECHARTS_STRATEGIST_GOAL_TEMPLATE = (
    'Based on the data analysis and user request ("{user_request}"), devise 1-3 suitable **ECharts** chart configurations '
    'for exploratory analysis. Focus on common EDA plots like histograms, scatter plots, bar charts, '
    'and box plots. Output ONLY the configurations as a list of valid JSON strings compatible with `echarts-for-react`. '
    'Each JSON object in the list should be a valid ECharts option object (containing `series`, `xAxis`, `yAxis`, etc.). '
    'Inside the `series` list items, use a "data_column" key to specify the primary column name for the series data. '
    'Inside the `xAxis` object, if it\'s categorical, use a "data_column" key to specify the column name for axis labels. '
    'Do NOT include the actual data values, only the column names.'
)
VISUALIZATION_CODER_GOAL_TEMPLATE = (
    'Based on the data analysis and user request ("{user_request}"), write Python code that uses Pandas, Plotly, and '
    'other libraries to process the data and create effective visualizations. Your code should be executable and '
    'generate high-quality, interactive visualizations that address the user\'s request. '
    'Focus on creating clean, efficient code that handles the data properly and produces visually appealing results. '
    'Output ONLY the Python code without any additional text or explanations.'
)
VISUALIZATION_EXECUTOR_GOAL_TEMPLATE = (
    'Execute the Python code provided by the Visualization Coder to generate visualizations for the user request: '
    '"{user_request}". Ensure the code runs correctly, handle any errors that arise, and convert the visualization '
    'outputs into formats compatible with the frontend (Plotly JSON and ECharts JSON configurations). '
    'Your goal is to bridge the gap between the code and the frontend visualization components.'
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
                        if "type" in parsed and "x" in parsed and "y" in parsed:
                            plotly_config = {
                                "data": [
                                    {
                                        "type": parsed.get("type", "scatter"),
                                        "x_column": parsed.get("x", "petal_length"),
                                        "y_column": parsed.get("y", "petal_width"),
                                        "mode": parsed.get("mode", "markers")
                                    }
                                ],
                                "layout": {
                                    "title": f"{parsed.get('y', 'petal_width')} vs {parsed.get('x', 'petal_length')}",
                                    "xaxis": {"title": parsed.get("x", "petal_length")},
                                    "yaxis": {"title": parsed.get("y", "petal_width")}
                                }
                            }
                            results.append(plotly_config)
                        else:
                            results.append(parsed)
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

                    # Try to create a default Plotly config if parsing fails
                    if "layout" in json_str and "data" in json_str:
                        try:
                            # Create a simple default Plotly config
                            default_config = {
                                "data": [
                                    {
                                        "type": "box",
                                        "x_column": "species",
                                        "y_column": "sepal_length"
                                    }
                                ],
                                "layout": {
                                    "title": "Distribution of Sepal Length by Species",
                                    "xaxis": {"title": "Species"},
                                    "yaxis": {"title": "Sepal Length"}
                                }
                            }
                            print("Created default Plotly config as fallback")
                            results.append(default_config)
                        except Exception as e:
                            print(f"Error creating default config: {e}")

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
def generate_visualizations(dataset_name: str, user_request: str) -> dict:
    """Instantiates and runs the CrewAI process with manual interpolation."""

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

    # 2. Instantiate Agents with formatted goals
    dataset_analyst = Agent(
        role='Data Analyst',
        goal=formatted_analyst_goal,
        backstory='An expert data analyst skilled in Pandas and exploratory data analysis (EDA). You excel at identifying key characteristics of datasets, understanding relationships between variables, and extracting meaningful insights from data. You are thorough in your analysis and can interpret both explicit and vague user requests to determine the most relevant aspects of the data to focus on.',
        verbose=True, llm=llm, tools=[dataset_info_tool], allow_delegation=False
    )
    plotly_strategist = Agent(
        role='Plotly Visualization Strategist',
        goal=formatted_plotly_goal,
        backstory='A specialist in data visualization using Plotly with years of experience creating effective visualizations for diverse datasets. You can translate any data insights and user requests into appropriate Plotly JSON configurations (`data` and `layout` objects). You are creative and can suggest multiple visualization approaches for the same data. You prioritize clarity, relevance, and visual appeal. You output ONLY the JSON list.',
        verbose=True, llm=llm, allow_delegation=False
    )
    echarts_strategist = Agent(
        role='ECharts Visualization Strategist',
        goal=formatted_echarts_goal,
        backstory='A specialist in data visualization using Apache ECharts with extensive experience in creating interactive and visually appealing charts. You excel at creating ECharts option objects that effectively represent data insights according to user needs. You are skilled at selecting the most appropriate chart types for different data patterns and user requests. You output ONLY the JSON list.',
        verbose=True, llm=llm, allow_delegation=False
    )
    visualization_coder = Agent(
        role='Visualization Coder',
        goal=formatted_coder_goal,
        backstory='An expert Python programmer specializing in data visualization with mastery of Pandas, Plotly, Matplotlib, Seaborn, and other visualization libraries. You write clean, efficient, and robust code that handles edge cases gracefully. You are skilled at data preprocessing and can create sophisticated visualizations that address both simple and complex user requests. You can interpret vague requests and implement appropriate visualizations.',
        verbose=True, llm=llm, allow_delegation=False
    )
    visualization_executor = Agent(
        role='Visualization Executor',
        goal=formatted_executor_goal,
        backstory='A technical expert who executes Python code to generate visualizations and ensures they are properly formatted for the frontend. You have deep knowledge of data visualization principles and can debug and optimize visualization code. You are resourceful in handling errors and can adapt code to work with different datasets and requirements. You bridge the gap between code and visual output, ensuring high-quality results.',
        verbose=True, llm=llm, tools=[python_code_execution_tool], allow_delegation=False
    )

    # 3. Instantiate Tasks (using placeholder-free descriptions/expected_outputs)
    # Ensure these strings do NOT contain {dataset_name} or {user_request}
    analyze_task = Task(
        description="1. Use the Dataset Information Tool to thoroughly examine the specified dataset. 2. Summarize the key findings (columns, types, distributions, correlations, interesting patterns) specifically focusing on aspects relevant to the user request provided in the goal. 3. For vague user requests like 'show some plots', identify the most interesting aspects of the data that would be worth visualizing. 4. Consider both obvious and non-obvious relationships in the data. 5. Provide a comprehensive analysis that will enable the visualization specialists to create effective and insightful visualizations.",
        expected_output="A detailed textual summary of the dataset analysis, highlighting columns, relationships, and characteristics relevant to the user request and suitable for visualization. For vague requests, include recommendations on what aspects of the data would be most interesting to visualize.",
        agent=dataset_analyst
    )
    plotly_task = Task(
        description="1. Review the data analysis summary and the original user request provided in the goal. 2. Identify 2 to 4 appropriate exploratory Plotly charts that would best represent the data and address the user's request. Consider a variety of chart types including: histograms, scatter plots, bar charts, box plots, heatmaps, line charts, bubble charts, and pie charts. 3. For each chart, determine the necessary columns from the specified dataset based on the analysis summary. 4. For vague requests like 'show some plots', create a diverse set of visualizations that highlight different aspects of the data. 5. Construct the Plotly JSON configuration list as specified in your goal (containing data and layout keys, using x_column/y_column). Ensure the JSON is valid and directly usable by `react-plotly.js`. Output ONLY the list of JSON objects.",
        expected_output="A list containing 2 to 4 valid Plotly JSON configuration objects (each having data and layout, using x_column/y_column). Output ONLY the list, nothing else.",
        agent=plotly_strategist,
        context=[analyze_task]
    )
    echarts_task = Task(
        description="1. Review the data analysis summary and the original user request provided in the goal. 2. Identify 2 to 4 appropriate exploratory ECharts charts that would best represent the data and address the user's request. Consider a variety of chart types including: bar charts, line charts, scatter plots, pie charts, radar charts, heatmaps, and tree maps. 3. For each chart, determine the necessary columns from the specified dataset based on the analysis summary. 4. For vague requests like 'show some plots', create a diverse set of visualizations that highlight different aspects of the data. 5. Construct the ECharts JSON option object list as specified in your goal (containing `series`, `xAxis`, `yAxis`, etc., using `data_column`). Ensure the JSON is valid and directly usable by `echarts-for-react`. Output ONLY the list of JSON objects.",
        expected_output="A list containing 2 to 4 valid ECharts JSON option objects. Output ONLY the list, nothing else.",
        agent=echarts_strategist,
        context=[analyze_task]
    )
    coder_task = Task(
        description="1. Review the data analysis summary and the original user request provided in the goal. 2. Write Python code that uses Pandas, Plotly, and other libraries to process the data and create effective visualizations. 3. Your code should load the dataset (which will be provided as 'df'), process it as needed, and create multiple visualizations (at least 3) that address the user's request. 4. For vague requests like 'show some plots', create a diverse set of visualizations that highlight different aspects of the data. 5. Include data preprocessing steps as needed (handling missing values, transformations, aggregations). 6. Use the 'save_plotly_fig(fig)' function to save any Plotly figures you create, and 'save_echarts_config(config)' for any ECharts configurations. 7. Output ONLY the Python code without any additional text or explanations.",
        expected_output="Python code that processes the data and creates multiple visualizations using Plotly and/or other libraries.",
        agent=visualization_coder,
        context=[analyze_task]
    )
    executor_task = Task(
        description="1. Review the Python code provided by the Visualization Coder. 2. Use the Python Code Execution Tool to execute the code with the specified dataset. 3. The tool will automatically handle loading the dataset and executing the code. 4. Review the execution results and ensure the visualizations are properly generated. 5. If there are any errors, fix them and re-execute the code. 6. If the code only generates one type of visualization, modify it to include at least one additional visualization type. 7. Ensure that the visualizations are diverse and highlight different aspects of the data. 8. Output the execution results, including all generated visualizations.",
        expected_output="Execution results including multiple generated visualizations in formats compatible with the frontend.",
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