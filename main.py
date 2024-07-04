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

if __name__ == '__main__':
    iris = datasets.load_iris()
    df = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )

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
    model, scaler, feature_names = classify_iris(X, y)
    new_data = pd.DataFrame([[5.1, 3.5, 5.0, 5.0]], columns=feature_names)

    row = scaler.transform(new_data)

    res = model.predict(row)
    print(f"The predicted iris class is: {res}")
