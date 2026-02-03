# Titanic Survival Prediction Project

## Project Group 9
**Members:**
- Jingtao Yang
- Yiran Wang
- Zihao Zhao

## Project Goal
Create a small yet performant model that predicts the survival of passengers aboard the Titanic. Emphasis is placed on building an efficient model that balances predictive performance with model simplicity.


## Dataset

Titanic dataset from [Data Science Dojo](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)

## Key Project Steps

1. **Data splitting**  
   - Split the dataset into training and test sets using 25% of the data as the test set.  

2. **Model selection**  
     - Logistic Regression  
     - K-Nearest Neighbors (KNN)   
     - Decision Trees  

3. **Model training and evaluation**  
   - Train models on the training set.  
   - Evaluate models on the test set and report performance metrics.  
   - Make comparisons between models.

## Project Structure

```
DATA-572-Group-9/
├── README.md                      
├── DATA 572 Project.pdf               # Project reqand 
|
├── titanic_augmented.csv              # Original augmented 
├── train_raw.csv                      # Training set 
├── test_raw.csv                       # Test set 
│
├── Drop_Columns_Split_Data.ipynb      # Initial data 
├── KNN.ipynb                          # K-Nearest Neighbors 
├── Logistic_Reg.ipynb                 # Logistic Regression 
├── Dec_Tree.ipynb                     # Decision Tree
├── Compare_three_models.ipynb         # Model comparison
|
├── tables/                            # Performance metrics 
│   ├── knn_cv_results.csv
│   ├── knn_test_performance.csv
│   ├── knn_feature_importance.csv
│   ├── logistic_cv_results.csv
│   ├── logistic_test_metrics.csv
│   ├── logistic_feature_importance.csv
│   ├── dt_results.csv
│   └── model_comparison.csv
│
└── plots/                             # Visualizations
    ├── knn_cv_accuracy.png
    ├── knn_confusion_matrix.png
    ├── knn_feature_importance_top15.png
    ├── logistic_cv_accuracy_vs_C.png
    ├── logistic_feature_contributions.png
    ├── dt_confusion_matrix.png
    └── model_comparison_accuracy.png
```

## Implementation Details

### Models Implemented

1. **K-Nearest Neighbors (KNN)**
   - Hyperparameters tuned: `n_neighbors`, `weights`
   - Best configuration: k=5, weights=uniform

2. **Logistic Regression**
   - Hyperparameters tuned: `C`, `class_weight`, `solver`

3. **Decision Tree**
   - Hyperparameters tuned: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `ccp_alpha`

### Data Preprocessing

- Numerical: median imputation + StandardScaler (KNN, Logistic Regression)
- Categorical: most-frequent imputation + one-hot encoding
- Dropped columns: PassengerId, Name, Ticket, Cabin, booking_reference, service_id


### Cross-Validation Strategy

- Stratified 5-Fold Cross-Validation
- Random State: 42 (for reproducibility)

## How to Run

1. Model Training and Evaluation
   ```
   Run in any order:
   - KNN.ipynb
   - Logistic_Reg.ipynb
   - Dec_Tree.ipynb
   ```

2. Model Comparison
   ```
   Run: Compare_three_models.ipynb
   Requires: All three models completed
   Output: Comparative analysis and visualizations
   ```

All notebooks use random_state=42 throughout to ensure deterministic results:
- Data splitting
- Cross-validation folds
- Model training (where applicable)
- Permutation importance


## Results Summary

### Model Performance Comparison

| Model | CV Best Accuracy | Test Accuracy |
|-------|------------------|---------------|
| **Logistic Regression** | 0.837 | 0.825 |
| **KNN** | 0.807 | 0.798 |
| **Decision Tree** | 0.811 | 0.785 |

*Note: See individual notebooks and `tables/model_comparison.csv` for detailed results.*

### Visualizations

Generated visualizations in `plots/`:
- Cross-validation performance curves
- Confusion matrices
- Feature importance/contribution plots
- Model comparison charts