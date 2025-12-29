# titanic-ahoy
# Titanic Survival Prediction with XGBoost

This project is a Machine Learning solution for the [Kaggle Titanic: Machine Learning from Disaster]((https://www.kaggle.com/competitions/titanic/overview)) competition. It utilizes **Feature Engineering** (specifically Title extraction and Age imputation) and uses an **XGBoost Classifier** optimized via **RandomizedSearchCV**.

## Project Overview

The goal is to predict passenger survival based on variables like age, sex, class, and family size. This solution improves upon baseline models by:
1.  **Extracting Titles** from passenger names to better estimate missing ages and social status.
2.  **Grouping Family Size** to capture the survival dependency of families.
3.  **Hyperparameter Tuning** to find the optimal settings for the Gradient Boosting model.

## Key Features

### 1. Feature Engineering
* **Title Extraction:** Parses the `Name` column to extract titles (Mr, Mrs, Master, etc.).
    * *Logic:* "Master" (boys) had much higher survival rates than "Mr" (adult men).
    * *Grouping:* Rare titles (Dr, Rev, Major) are grouped into a single "Rare" category. French titles (Mlle, Mme) are mapped to their English equivalents.
* **Smart Age Imputation:** Instead of using the global average age, missing ages are filled using the **median age of the passenger's specific Title group**.
* **Family Size:** Combines `SibSp` (siblings/spouses) and `Parch` (parents/children) to create a total `Family_Size` feature.

### 2. Model & Tuning
* **Algorithm:** `XGBClassifier` (Extreme Gradient Boosting).
* **Optimization:** `RandomizedSearchCV` is used to search for the best hyperparameters across 50 iterations.
* **Reproducibility:** A fixed `random_state` (55) is used for both the model and the splitter to ensure consistent results.

## Performance
My Model had a prediction score of **0.77751** and a rank of **5468(out of 15724)** as of 29th December 2025.
## Dependencies

To run this code locally, you need the following Python libraries:

```python
pip install pandas numpy xgboost scikit-learn scipy
