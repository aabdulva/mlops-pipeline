import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data():
    df = pd.read_csv('data/iris.csv')
    
    # Clean numeric columns
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.]', '', regex=True), errors='coerce')
    
    # Clean species names
    df['species'] = df['species'].str.strip().str.lower()
    valid_species = {'setosa', 'versicolor', 'virginica'}
    df = df[df['species'].isin(valid_species)]  # Filter valid species
    
    # Check class balance
    print("Class distribution:\n", df['species'].value_counts())
    
    # Encode labels
    le = LabelEncoder()
    df['species'] = le.fit_transform(df['species'])
    
    # Verify at least 2 classes
    if len(df['species'].unique()) < 2:
        raise ValueError("Dataset must contain at least 2 classes")
    
    return train_test_split(df.drop('species', axis=1), df['species'], test_size=0.2, random_state=42)

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    
    print("\nFeature types:")
    print(X_train.dtypes)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.2f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/iris_classifier.joblib')
    return accuracy

if __name__ == "__main__":
    train_model()
    