# Databricks notebook source
# MAGIC %md
# MAGIC # Model Step 0: Import

# COMMAND ----------

import numpy as np
import warnings
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score , classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.metrics import precision_score, accuracy_score
from sklearn.feature_selection import RFE
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime
import os

# COMMAND ----------

seed = 250

# COMMAND ----------

warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Step 1: Read Data

# COMMAND ----------

X_train_transformed_loaded = np.load('raw_data/X_train_transformed.npy')
X_val_transformed_loaded = np.load('raw_data/X_val_transformed.npy')

y_train = np.load('raw_data/y_train.npy')
y_val = np.load('raw_data/y_eval.npy')

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Step 2: Pipeline
# MAGIC

# COMMAND ----------

rf_pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=seed)),
    ('classifier', RandomForestClassifier(random_state=seed))
])

lr_pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=seed)),
    ('classifier', LogisticRegression(random_state=seed))
])

# COMMAND ----------

voting_clf = VotingClassifier(estimators=[
    ('rf', rf_pipeline),
    ('lr', lr_pipeline)
], voting='soft')

voting_clf.fit(X_train_transformed_loaded, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Step 3: Evaluation

# COMMAND ----------

# Evaluate the ensemble model and generate confusion matrices
y_pred_prob = voting_clf.predict_proba(X_val_transformed_loaded)[:, 1]

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

# COMMAND ----------

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')
display.plot()

# COMMAND ----------

joblib.dump(voting_clf, 'model_history/classifier-v0.0.2.joblib')
