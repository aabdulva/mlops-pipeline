# mlops-pipeline
Study project for MLOPS
# MLOps Pipeline with GitHub Actions

This project demonstrates a complete MLOps pipeline with:
- Automated training and testing of a logistic regression model
- CI/CD pipelines using GitHub Actions
- Docker containerization for deployment

## Features

- **Automated Training**: Model is trained on push/pull request
- **Model Testing**: Accuracy tests ensure model quality
- **CI/CD Pipeline**: Separate workflows for continuous integration and deployment
- **Docker Support**: Ready for containerized deployment

## Workflows

1. **CI Pipeline**:
   - Runs on push/pull request
   - Trains the model
   - Tests model accuracy
   - Uploads model as artifact

2. **CD Pipeline**:
   - Triggers after successful CI
   - Tests deployment
   - Builds and pushes Docker image (on main branch)

## How to Use

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train model: `python src/train.py`
4. Test model: `python src/test_model.py`
5. Make prediction: `python src/predict.py`

## GitHub Actions

The workflows are defined in `.github/workflows/`:
- `ci.yml`: Continuous Integration
- `cd.yml`: Continuous Deployment

## Results
[![CI Pipeline](https://github.com/aabdulva/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/aabdulva/mlops-pipeline/actions)
[![CD Pipeline](https://github.com/aabdulva/mlops-pipeline/actions/workflows/cd.yml/badge.svg)](https://github.com/aabdulva/mlops-pipeline/actions)