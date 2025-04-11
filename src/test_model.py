import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def test_model():
    # Load model AND label encoder
    model = joblib.load('models/iris_classifier.joblib')
    le = LabelEncoder()
    
    # Load and preprocess data (MUST match training exactly)
    df = pd.read_csv('data/iris.csv')
    
    # 1. Clean numeric columns
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('[^0-9.]', '', regex=True), 
            errors='coerce'
        )
    
    # 2. Clean species names
    df['species'] = df['species'].str.strip().str.lower()
    valid_species = {'setosa', 'versicolor', 'virginica'}
    df = df[df['species'].isin(valid_species)].dropna()
    
    # 3. Encode labels SAME WAY AS TRAINING
    le.fit(['setosa', 'versicolor', 'virginica'])  # Force same encoding order
    y_test = le.transform(df['species'])
    
    X_test = df.drop('species', axis=1)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy >= 0.7, f"Model accuracy {accuracy} is below threshold 0.7"
    print(f"Test passed with accuracy: {accuracy}")

if __name__ == "__main__":
    test_model()