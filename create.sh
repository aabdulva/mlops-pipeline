# Create local directory structure
mkdir -p ./{.github/workflows,data,models,src}

# Create files
touch ./{.github/workflows/{ci.yml,cd.yml},data/iris.csv,models/.gitkeep,src/{train.py,predict.py,test_model.py},requirements.txt,Dockerfile,README.md}

# Initialize git repo
cd mlops-pipeline
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit with project structure"

# Connect to GitHub (after creating empty repo on GitHub)
git remote add origin https://github.com/aabdulva/mlops-pipeline.git
git branch -M main
git push -u origin main