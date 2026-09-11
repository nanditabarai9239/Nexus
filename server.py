import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import ML Tools
from src.tools.ml_tool import make_prediction_dict
from src.agent import get_agent
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Nexus - Advanced Predictive Decision Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class PredictRequest(BaseModel):
    customer_data: dict

class ChatRequest(BaseModel):
    message: str
    history: list[dict]

class InsightRequest(BaseModel):
    var1: str
    var2: str

def is_api_key_valid():
    return bool(os.getenv("GOOGLE_API_KEY"))

def get_models_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

def generate_llm_insight(prompt: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2,
        convert_system_message_to_human=True
    )
    return llm.invoke(prompt).content

# API Routes
@app.get("/api/stats")
async def get_stats():
    """Returns dynamic dataset statistics, feature importance, and advanced metrics."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'customer_churn.csv')
    models_dir = get_models_dir()
    
    if not os.path.exists(data_path) or not os.path.exists(os.path.join(models_dir, 'target_col.pkl')):
        raise HTTPException(status_code=404, detail="Dataset or model not found")
        
    df = pd.read_csv(data_path)
    target_col = joblib.load(os.path.join(models_dir, 'target_col.pkl'))
    
    # Advanced Metrics
    metrics = joblib.load(os.path.join(models_dir, 'metrics.pkl'))
    
    # Feature importances for chart
    fi_dict = joblib.load(os.path.join(models_dir, 'feature_importances.pkl'))
    top_features = dict(sorted(fi_dict.items(), key=lambda item: item[1], reverse=True)[:10])
    
    # Target distribution
    target_dist = df[target_col].value_counts().to_dict()
    
    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "target_col": target_col,
        "api_key_valid": is_api_key_valid(),
        "top_features": top_features,
        "target_dist": target_dist,
        "metrics": metrics
    }

@app.get("/api/data_quality")
async def get_data_quality():
    """Returns EDA and data quality insights."""
    models_dir = get_models_dir()
    try:
        data_quality = joblib.load(os.path.join(models_dir, 'data_quality.pkl'))
        corr_matrix = joblib.load(os.path.join(models_dir, 'correlation_matrix.pkl'))
        return {
            "quality": data_quality,
            "correlation": corr_matrix
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="Data Quality report not generated yet.")

@app.get("/api/schema")
async def get_schema():
    """Returns the dynamically generated schema for the prediction form."""
    models_dir = get_models_dir()
    schema_path = os.path.join(models_dir, 'schema.pkl')
    
    if not os.path.exists(schema_path):
        raise HTTPException(status_code=404, detail="Schema not found")
        
    schema = joblib.load(schema_path)
    target_col = joblib.load(os.path.join(models_dir, 'target_col.pkl'))
    
    return {
        "target_col": target_col,
        "schema": schema
    }

@app.post("/api/insights")
async def generate_insights(req: InsightRequest):
    """Generates dynamic AI insights based on two selected variables."""
    if not is_api_key_valid():
        raise HTTPException(status_code=401, detail="API Key not configured on the backend.")
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'customer_churn.csv')
    models_dir = get_models_dir()
    
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        df = pd.read_csv(data_path)
        target_col = joblib.load(os.path.join(models_dir, 'target_col.pkl'))
        
        # Calculate summary data for the chart
        # Keep it simple: group by var1, aggregate var2 (count if cat, mean if num)
        # Or just return value counts if they are both categorical
        # For simplicity, let's just use crosstab or groupby depending on types
        
        is_num_var1 = pd.api.types.is_numeric_dtype(df[req.var1])
        is_num_var2 = pd.api.types.is_numeric_dtype(df[req.var2])
        
        chart_data = {"labels": [], "values": []}
        summary_text = ""
        
        if not is_num_var1 and is_num_var2:
            grouped = df.groupby(req.var1)[req.var2].mean().reset_index()
            chart_data["labels"] = grouped[req.var1].astype(str).tolist()
            chart_data["values"] = grouped[req.var2].tolist()
            summary_text = grouped.to_string(index=False)
        elif is_num_var1 and not is_num_var2:
            grouped = df.groupby(req.var2)[req.var1].mean().reset_index()
            chart_data["labels"] = grouped[req.var2].astype(str).tolist()
            chart_data["values"] = grouped[req.var1].tolist()
            summary_text = grouped.to_string(index=False)
        else:
            # Both categorical or both numeric -> just do value counts of var1 to show distribution
            counts = df[req.var1].value_counts().head(10)
            chart_data["labels"] = counts.index.astype(str).tolist()
            chart_data["values"] = counts.values.tolist()
            summary_text = df[[req.var1, req.var2, target_col]].describe(include='all').to_string()
            
        # Get AI Insight
        prompt = f"""
You are a senior data analyst. You are looking at a dataset where the target variable is '{target_col}'.
The user wants an insight about the relationship between '{req.var1}' and '{req.var2}'.

Here is a brief statistical summary of these variables:
{summary_text}

Write a short, professional, and actionable business insight (1-2 paragraphs) about how these factors might influence '{target_col}'. 
Do not use generic fluff. Use specific numbers from the summary if available. Format with HTML paragraphs (<p>).
        """
        
        insight_html = generate_llm_insight(prompt)
        
        return {
            "chart_data": chart_data,
            "insight_html": insight_html,
            "var1": req.var1,
            "var2": req.var2
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict(req: PredictRequest):
    """Makes a prediction based on generic profile."""
    try:
        result = make_prediction_dict(req.customer_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Interacts with the LangChain Agent."""
    if not is_api_key_valid():
        raise HTTPException(status_code=401, detail="API Key not configured on the backend.")
        
    try:
        agent = get_agent()
        # Format history as text for context
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in req.history[-5:]])
        prompt = f"Context history:\n{history_text}\n\nUser Question: {req.message}"
        
        response = agent.invoke({"input": prompt})
        bot_reply = response.get("output", str(response))
        
        return {"reply": bot_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...), 
    target_col: str = Form(...), 
    strategy: str = Form("fast")
):
    """Uploads a new dataset and retrains the model using AutoML or Fast strategy."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'customer_churn.csv')
    
    contents = await file.read()
    with open(data_path, 'wb') as f:
        f.write(contents)
        
    # Retrain model
    from src.train_model import train_and_evaluate
    models_dir = os.path.join(base_dir, 'models')
    try:
        train_and_evaluate(data_path, models_dir, target_col, strategy=strategy)
        return {"status": "success", "message": f"Dataset uploaded and model retrained ({strategy} mode)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrain model: {str(e)}")

@app.get("/api/export_model")
async def export_model():
    """Download the trained best model .pkl file."""
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, 'best_model.pkl')
    if os.path.exists(model_path):
        return FileResponse(model_path, media_type='application/octet-stream', filename='nexus_best_model.pkl')
    else:
        raise HTTPException(status_code=404, detail="Model file not found.")

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=True)
