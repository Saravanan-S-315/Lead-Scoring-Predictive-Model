import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Load the Lead dataset."""
    try:
        leads = pd.read_csv("Lead.csv")
        return leads
    except FileNotFoundError:
        print("Error: Lead.csv not found.")
        return None

def clean_data(leads):
    """Perform data cleaning and handling missing values."""
    # Drop IDs
    leads.drop(columns=['Prospect ID', 'Lead Number'], axis=1, inplace=True, errors='ignore')
    
    # Replace 'Select' with NaN
    leads = leads.replace("Select", np.nan)
    
    # Dropping columns with > 40% missing values
    leads = leads.loc[:, leads.isnull().mean() < 0.4]
    
    # Impute or drop remaining nulls for numerical visits
    leads = leads.dropna(subset=["TotalVisits", "Page Views Per Visit"])
    return leads

def evaluate_model(y_actual, y_prob, cutoff=0.3):
    """Generate Lead Scores and binary predictions."""
    y_pred = y_prob.map(lambda x: 1 if x > cutoff else 0)
    score = (y_prob * 100).round()
    
    print(f"\n--- Model Results (Cutoff: {cutoff}) ---")
    print(f"Accuracy: {metrics.accuracy_score(y_actual, y_pred):.2f}")
    print(f"Precision: {metrics.precision_score(y_actual, y_pred):.2f}")
    print(f"Recall: {metrics.recall_score(y_actual, y_pred):.2f}")
    return score

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = clean_data(df)
        
        # Binary Mapping
        bin_cols = ["Do Not Email", "Do Not Call", "Search", "Magazine", "Newspaper Article"]
        for col in bin_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})
        
        # Categorical Encoding
        cat_cols = df.select_dtypes(include='object').columns
        leads_final = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # Train-Test Split
        X = leads_final.drop("Converted", axis=1)
        y = leads_final["Converted"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
        
        # Scaling
        scaler = StandardScaler()
        num_cols = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        
        # Model
        lr = LogisticRegression()
        rfe = RFE(lr, n_features_to_select=15)
        rfe.fit(X_train, y_train)
        
        # Output Logic
        print("Model Trained Successfully.")
        # [Remaining logic for test set evaluation...]
