# liver_disease_predictor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# Load the dataset
data = pd.read_csv("data/indian_liver_patient.csv")

# Preprocessing
data = data.dropna()  # Remove rows with missing values
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])  # Encode Gender (1 = Male, 0 = Female)

# Define features and target
X = data.drop('Dataset', axis=1)  # Assuming 'Dataset' column is the target
y = data['Dataset']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Test the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Save the model and scaler
with open("models/knn_model.pkl", "wb") as model_file:
    pickle.dump(knn, model_file)

with open("models/scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
