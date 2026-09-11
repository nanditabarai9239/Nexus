# Nexus - Predictive Decision Platform

Nexus is an enterprise-grade AI intelligence platform designed to transform raw tabular data into actionable business insights. It bridges the gap between complex machine learning and business operations by providing a clean, no-code interface for data analysis, predictive modeling, and generative AI insights.

## Core Features

- **Automated Machine Learning (AutoML)**: Upload your dataset (CSV) and select a target variable. Nexus automatically handles data imputation, categorical encoding, scaling, and trains multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting) to find the best performing model.
- **Dynamic AI Insights**: Select any two variables from your dataset and instantly receive a visual correlation graph paired with a written, AI-generated business insight explaining the relationship and its potential impact on your target outcome.
- **Business-Readable Inference**: Input a new data profile and receive clear, human-readable predictions (e.g., "🚨 High Risk of Positive Outcome" or "✅ Low Risk") instead of raw probabilities, complete with model confidence scores.
- **Enterprise UI**: A clean, responsive, and professional dashboard designed for clarity and ease of use, moving away from cluttered or overly "techy" aesthetics.
- **Interactive Data Explorer**: View automated Exploratory Data Analysis (EDA) including dataset dimensions, missing values, and a dynamic correlation heatmap.
- **Integrated AI Assistant**: Chat directly with an LLM agent that has context on your dataset's schema and model metrics to help answer questions about your data.

## Technology Stack

- **Backend**: Python, FastAPI
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Generative AI**: Google Gemini 3.5 Flash (via LangChain)
- **Frontend**: Vanilla HTML/CSS/JavaScript, Chart.js

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd predictive_decision_platform
   ```

2. **Set up a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API Key:**
   - Create a `.env` file in the root directory (you can copy `.env.example`).
   - Add your Gemini API key:
     ```env
     GEMINI_API_KEY=your_api_key_here
     ```

5. **Run the application:**
   ```bash
   python server.py
   ```
   The application will be available at `http://localhost:8501`.

## Usage
1. **Upload**: Navigate to the upload section, provide a CSV dataset, select your target column, and click "Confirm Upload". 
2. **Explore**: Use the Dashboard and Data Explorer tabs to review model accuracy and feature correlations.
3. **Analyze**: Use the Dynamic Insights tab to compare variables and generate AI reports.
4. **Predict**: Use the Predictive Model tab to input a new record and get a real-time risk assessment.
