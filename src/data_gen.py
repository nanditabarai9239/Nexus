import pandas as pd
import numpy as np
import os

def generate_telecom_churn_data(n_samples=2000, output_dir='data'):
    """
    Generates a realistic synthetic dataset for telecom customer churn.
    """
    np.random.seed(42)
    
    # Feature generation
    customer_ids = [f'CUST_{i:04d}' for i in range(n_samples)]
    
    # Demographics
    age = np.random.normal(45, 15, n_samples).astype(int)
    age = np.clip(age, 18, 80)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Account information
    tenure_months = np.random.randint(1, 72, n_samples)
    contract_type = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 
        n_samples, 
        p=[0.55, 0.25, 0.20]
    )
    monthly_charges = np.random.uniform(20.0, 120.0, n_samples)
    
    # Usage metrics
    total_data_gb = np.random.exponential(50, n_samples)
    customer_service_calls = np.random.poisson(1.5, n_samples)
    
    # Introduce some correlations with churn
    # Higher churn probability for: Month-to-month, high service calls, short tenure, high charges
    churn_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.1 # Base probability
        
        if contract_type[i] == 'Month-to-month':
            prob += 0.3
        if customer_service_calls[i] > 3:
            prob += 0.2
        if tenure_months[i] < 12:
            prob += 0.15
        if monthly_charges[i] > 80:
            prob += 0.1
            
        churn_prob[i] = min(prob, 0.95) # Cap at 95%
        
    # Generate target variable based on probabilities
    churn = np.random.binomial(1, churn_prob)
    churn_labels = ['Yes' if c == 1 else 'No' for c in churn]
    
    # Total charges (approximate)
    total_charges = monthly_charges * tenure_months * np.random.uniform(0.9, 1.1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': age,
        'Gender': gender,
        'TenureMonths': tenure_months,
        'ContractType': contract_type,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'TotalDataGB': np.round(total_data_gb, 2),
        'CustomerServiceCalls': customer_service_calls,
        'Churn': churn_labels
    })
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'customer_churn.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset with {n_samples} records saved to {output_path}")
    
if __name__ == "__main__":
    # Go up one directory if run from src/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    generate_telecom_churn_data(output_dir=data_dir)
