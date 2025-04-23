import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

'''
pip install pandas scikit-learn joblib
'''

# Load the dataset
# first unpack heart.csv.tar.bz
df = pd.read_csv("heart.csv")

# Define numerical columns (matching config.json)
numerical_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# Initialize and fit the scaler
scaler = StandardScaler()
scaler.fit(df[numerical_cols])

# Save the scaler
dump(scaler, "scaler.joblib")
print("scaler.joblib created successfully.")