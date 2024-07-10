# Databricks notebook source
# MAGIC %md
# MAGIC # Model Step 0: Import

# COMMAND ----------

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning models and utilities
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, classification_report, precision_recall_curve, roc_curve, auc, precision_score, RocCurveDisplay
from sklearn.feature_selection import RFECV, RFE, SelectKBest, f_classif


# Imbalanced dataset handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Model persistence
import joblib

# Deep learning
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Experiment tracking
import mlflow

# Miscellaneous utilities
import datetime
import os
import warnings

# COMMAND ----------

seed = 250

# COMMAND ----------

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Step 1: Read Data

# COMMAND ----------

X_train_transformed_loaded = np.load('X_train_transformed.npy')
X_val_transformed_loaded = np.load('X_val_transformed.npy')

y_train = np.load('y_train.npy')
y_val = np.load('y_eval.npy')

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Step 2: Pipeline
# MAGIC

# COMMAND ----------

logistic_pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=seed)),
    ('classifier', LogisticRegression(random_state=seed))
])
param_grid = {
        'classifier__C': [0.1, 13, 15, 20, 40],
        'classifier__max_iter': [500, 1000],
        'classifier__penalty': ['l2'],
        'classifier__class_weight': ['balanced']
}
cv = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(estimator=logistic_pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_transformed_loaded, y_train)
best_model = grid_search.best_estimator_

# COMMAND ----------

grid_search.best_params_

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Step 3: Evaluation

# COMMAND ----------

with mlflow.start_run(run_name="exp-02-logistics"):
    # Evaluate the ensemble model and generate confusion matrices
    y_pred_prob = best_model.predict_proba(X_val_transformed_loaded)[:, 1]

    # Adjust the decision threshold
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)
    f1_scores = (2*recall + precision)/(recall+precision)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]

    adjusted_threshold = 0.53

    y_pred = (y_pred_prob >= adjusted_threshold).astype(int)

    cm = confusion_matrix(y_val, y_pred)
    cr = classification_report(y_val, y_pred)
    results = {
        'best_threshold': best_threshold,
        'confusion_matrix': cm,
        'classification_report': cr
    }

    # Print results
    print("Ensemble Model")
    print(f"Best Decision Threshold: {results['best_threshold']:.2f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Log model
    mlflow.sklearn.log_model(best_model, "logistics_regression")
    
    # Log parameters, metrics, or any other information if needed
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("n", 18)

    mlflow.log_metric("validation_precision", precision_score(y_val, y_pred))
    mlflow.log_metric("validation_recall", recall_score(y_val, y_pred))
    mlflow.log_metric("validation_accuracy", accuracy_score(y_val, y_pred))
    
    # End the MLflow run
    mlflow.end_run()



# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')
display.plot()

# COMMAND ----------

joblib.dump(best_model, 'classifier-v0.0.1.joblib')