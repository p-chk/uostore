# Databricks notebook source
# MAGIC %md
# MAGIC # Transform Step 0: Import

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.base import TransformerMixin 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform Step 1: Read Data

# COMMAND ----------

train_df = pd.read_csv('raw_data/train_df.csv')
train_df = train_df.drop(columns=['Unnamed: 0'])
train_df = train_df[train_df.Year_Birth>1923]

# COMMAND ----------

eval_df = pd.read_csv('raw_data/eval_df.csv')
eval_df = eval_df.drop(columns=['Unnamed: 0'])
eval_df = eval_df[eval_df.Year_Birth>1923]

# COMMAND ----------

X_train = train_df.drop(columns=['Response'])
y_train = train_df['Response']
X_eval = eval_df.drop(columns=['Response'])
y_eval = eval_df['Response']

# COMMAND ----------

# MAGIC %md
# MAGIC # Transform Step 2: Transform Pipeline

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

numeric_features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',  'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
categorical_features = ['Education', 'Marital_Status', 'Complain']


# Pipelines for numerical and categorical transformations
numeric_transformer = Pipeline(steps=[
    ('outlier_remover', OutlierRemover()),
    ('imputer', SimpleImputer(strategy='median'))])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label_ec', LabelEC())])

# Combine into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)],
    remainder='drop')

# COMMAND ----------

# Create the final pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline on the training data
pipeline.fit(pd.concat((X_train, X_eval)),pd.concat((y_train, y_eval)))

# Transform the training and validation data
X_train_transformed = pipeline.transform(X_train)
X_eval_transformed = pipeline.transform(X_eval)

print("Transformed Training Data:\n", X_train_transformed)
print("\nTransformed Validation Data:\n", X_eval_transformed)

# COMMAND ----------

# Save the pipeline
joblib.dump(pipeline, 'preprocess_pipeline/data_preprocessor_pipeline.pkl')

# Save the transformed datasets
np.save('raw_data/X_train_transformed.npy', X_train_transformed)
np.save('raw_data/X_val_transformed.npy', X_eval_transformed)
np.save('raw_data/y_train.npy', y_train)
np.save('raw_data/y_eval.npy', y_eval)
