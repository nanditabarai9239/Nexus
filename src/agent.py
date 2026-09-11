import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from src.tools.ml_tool import run_inference
from langchain.tools import tool

# Load data for the data tool
def get_data_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', 'customer_churn.csv')

@tool
def get_dataset_summary(query: str = "") -> str:
    """Returns a summary of the customer churn dataset, including columns, data types, and basic statistics. Always use this to understand the data before querying."""
    try:
        df = pd.read_csv(get_data_path())
        desc = df.describe(include='all').to_string()
        return f"Dataset has {len(df)} rows. Here is the summary:\n{desc}"
    except Exception as e:
        return f"Error loading data: {e}"

@tool
def query_average_metrics(segment_column: str, metric_column: str) -> str:
    """
    Calculates the average of a metric column grouped by a segment column.
    Example: segment_column='Gender', metric_column='MonthlyCharges'
    Check the summary first to know what columns exist.
    """
    try:
        df = pd.read_csv(get_data_path())
        if segment_column not in df.columns or metric_column not in df.columns:
            return f"Invalid columns. Available: {', '.join(df.columns)}"
            
        result = df.groupby(segment_column)[metric_column].mean().reset_index()
        return f"Average {metric_column} by {segment_column}:\n{result.to_string(index=False)}"
    except Exception as e:
        return f"Error querying data: {e}"

def get_agent():
    """Initializes and returns the LangChain agent executor."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0,
        convert_system_message_to_human=True
    )
    
    tools = [run_inference, get_dataset_summary, query_average_metrics]
    
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor
