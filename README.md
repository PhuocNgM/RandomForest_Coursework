# RandomForest_Coursework
## Overview

This repository contains a small coursework project implementing a Random Forest classifier from scratch and comparing it with scikit-learn's RandomForestClassifier on the Sonar dataset. It includes:
- Custom implementation: module.py (CustomRandomForestClassifier)
- Utilities: util.py (simple cross_val_score)
- Reproducible run script: main.py
- Jupyter notebook: notebook.ipynb
- Environment helper: create_env.sh
- Dependencies: requirements.txt

## Quickstart — recommended (Unix / macOS / WSL)

1. Create and activate the virtual environment and install dependencies:
```
./create_env.sh
source python_env/bin/activate
```

The script will:
- create (or reuse) a venv in `python_env/`
- install packages from `requirements.txt`
- register an IPython kernel named `randomforest-coursework`

2. Run the comparison script:
```
python main.py
```
This script downloads (via kagglehub) or uses the included `data/sonar/sonar.all-data.csv`, trains both scikit-learn and the custom Random Forest for different numbers of trees, and prints per-fold accuracies.

3. (Optional) Launch the Jupyter notebook:
```
jupyter notebook notebook.ipynb
# or
jupyter lab
```
Open `notebook.ipynb` to interactively explore the dataset and experiments.