# DVC tracks large data files and model artifacts
# that shouldn't be in Git directly.
# Guideline: Version Control - Git LFS, DVC

# To initialize DVC:
#   dvc init
#   dvc add data/raw/creditcard.csv
#   dvc add models/best_model.joblib
#   git add data/raw/.gitkeep data/raw/creditcard.csv.dvc models/best_model.joblib.dvc
#   git commit -m "Track data and model with DVC"

# To pull data on a new machine:
#   dvc pull
