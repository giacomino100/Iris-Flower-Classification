import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global variables to store the model, scaler, and feature names
global_model = None
global_scaler = None
global_feature_names = None

def check_duplicated(df):
    duplicates = df.duplicated()
    print(f'Number of duplicate rows: {duplicates.sum()}')

def pre_process_data(df):
    # Check for missing values
    print(f'Missing values\n ---------------\n{df.isnull().sum()}')

    # Fill missing values (if any)
    df.fillna(df.mean(), inplace=True)

    #remove duplicate
    df.drop_duplicates(inplace=True)

    return df

def classify_iris(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    return model, scaler, X.columns.tolist()


@app.route('/predict', methods=['POST'])
def predict():
    global global_model, global_scaler, global_feature_names

    data = request.json
    new_data = pd.DataFrame([data], columns=global_feature_names)

    row = global_scaler.transform(new_data)
    prediction = global_model.predict(row)

    return jsonify({
        "prediction": int(prediction[0]),
        "class_name": iris.target_names[int(prediction[0])]
    })

if __name__ == '__main__':
    iris = datasets.load_iris()
    df = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )

    print(iris.target_names)
    #df.info(): Get information about the DataFrame
    #df.describe(): Get statistical summary of numerical columns
    #df.columns: See column names
    #df.shape: Get the dimensions of the DataFrame

    check_duplicated(df)

    df = pre_process_data(df)

    # Prepare features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']

    # Perform classification
    global_model, global_scaler, global_feature_names = classify_iris(X, y)

    app.run(debug=True)

