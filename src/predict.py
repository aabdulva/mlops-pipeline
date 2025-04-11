import joblib
import pandas as pd

model = joblib.load('models/iris_classifier.joblib')

def predict(features):
    return model.predict([features])[0]

if __name__ == "__main__":
    # Example prediction
    sample = [5.1, 3.5, 1.4, 0.2]  # Should predict 'setosa'
    print(f"Prediction for {sample}: {predict(sample)}")