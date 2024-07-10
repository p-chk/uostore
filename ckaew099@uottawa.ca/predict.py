# Databricks notebook source
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
from pyspark.sql import SparkSession

# COMMAND ----------

dbutils.widgets.text("version", "v0.0.1", "Version to serve")
dbutils.widgets.text("column_name", "success_rate_v0.0.1", "Name of the to-predict")
version = dbutils.widgets.get("version")
column_name = dbutils.widgets.get("column_name")

# COMMAND ----------

model = joblib.load(f'model_history/classifier-{version}.joblib')

# COMMAND ----------

serving_df = np.load('raw_data/X_serving_transformed.npy')


# COMMAND ----------

y_pred_prob = model.predict_proba(serving_df)[:, 1]

# COMMAND ----------

y_pred_prob

# COMMAND ----------

customer_pd = spark.read.table('`customer`')
customer_pd = customer_pd.toPandas()
customer_pd[column_name] = y_pred_prob.tolist()
customer_with_pred = spark.createDataFrame(customer_pd)

customer_with_pred.write.mode('overwrite').option("mergeSchema", "true").saveAsTable('customer')
