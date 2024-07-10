# Databricks notebook source
# MAGIC %md
# MAGIC # Transform Serving Step 0: Import

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform Serving Step 1: Read Data

# COMMAND ----------

serving_df = spark.read.table('`customer`')
serving_df = serving_df.toPandas()

# COMMAND ----------

serving_df = serving_df[['Id', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
       'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth','Complain']]

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform Serving Step 2: Pipeline

# COMMAND ----------

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for column in X.columns:
            if np.issubdtype(X[column].dtype, np.number):
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                X[column] = np.where(X[column] < lower_bound, lower_bound, X[column])
                X[column] = np.where(X[column] > upper_bound, upper_bound, X[column])
        return X

# COMMAND ----------

class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reference_date=None):
        self.reference_date = reference_date if reference_date is not None else pd.Timestamp('today')
    
    def fit(self, X_train, y=None):
        return self
    
    def transform(self, X_train):
        X_train = X_train.copy()
        X_train['Dt_Customer'] = (self.reference_date - X_train['Dt_Customer']).dt.days
        return X_train

# COMMAND ----------

class LabelEC(TransformerMixin):
    encoder = {}
    def __init__(self, *args, **kwargs):
        self.encoder = {}

    def fit(self, x, y=0):
        temp_categorical = pd.DataFrame(x)
        for (columnName, columnData) in temp_categorical.iteritems():
            new_encoder = LabelEncoder()
            new_encoder.fit(columnData)
            self.encoder[columnName] = new_encoder
        return self

    def transform(self, x, y=0):
        temp_categorical = pd.DataFrame(x)
        for (columnName, columnData) in temp_categorical.iteritems():
            temp_categorical[columnName] = self.encoder[columnName].transform(columnData)
        return temp_categorical.to_numpy()

# COMMAND ----------

# Load the pipeline
loaded_pipeline = joblib.load('data_preprocessor_pipeline.pkl')

X_serving_transformed = loaded_pipeline.transform(serving_df)
np.save('X_serving_transformed.npy', X_serving_transformed)