# Databricks notebook source
# MAGIC %md
# MAGIC # Validation Step 0: Import

# COMMAND ----------

pip install tensorflow-data-validation==1.8.0

# COMMAND ----------

# Import packages
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tempfile, urllib, zipfile
import tensorflow_data_validation as tfdv


from tensorflow.python.lib.io import file_io
from tensorflow_data_validation.utils import slicing_util
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList, DatasetFeatureStatistics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC


# Set TF's logger to only display errors to avoid internal warnings being shown
tf.get_logger().setLevel('ERROR')

# COMMAND ----------

seed=250

# COMMAND ----------

# MAGIC %md
# MAGIC # Validation Step 1: Read Customer History

# COMMAND ----------

df = spark.read.table('`customer_history`')

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Validation Step 2: Split Data

# COMMAND ----------

def prepare_data_splits_from_dataframe(df, target):
    '''
    Splits a Pandas Dataframe into training, evaluation and serving sets.

    Parameters:
            df : pandas dataframe to split

    Returns:
            train_df: Training dataframe(70% of the entire dataset)
            eval_df: Evaluation dataframe (15% of the entire dataset) 
            serving_df: Serving dataframe (15% of the entire dataset, label column dropped)
    '''

    df = df.dropna()
    df['Dt_Customer'] = df['Dt_Customer'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else x)
    X = df.drop(columns=[target])
    y = df[target]

    X_t, X_serve, y_t, y_serve = train_test_split(X, y, test_size=0.1, random_state=seed, stratify=y)

    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=seed, stratify=y_t)

    train_df = pd.concat((X_train, y_train), axis=1)
    
    test_df = pd.concat((X_test, y_test), axis=1)
    # Serving data emulates the data that would be submitted for predictions, so it should not have the label column.
    serving_df = X_serve

    return train_df, test_df, serving_df

# COMMAND ----------

# Split the datasets
train_df, eval_df, serving_df = prepare_data_splits_from_dataframe(df, 'Response')
print('Training dataset has {} records\nValidation dataset has {} records\nServing dataset has {} records'.format(len(train_df),len(eval_df),len(serving_df)))

# COMMAND ----------

# Feature selection function
def transform(X, y, k=4):
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    X_new_df = pd.DataFrame(X_new, columns=X.columns[selected_features])

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new_df)

    return pd.DataFrame(X_scaled, columns=X_new_df.columns), selector

# COMMAND ----------

# MAGIC %md
# MAGIC # Validation Step 3: Validate Training

# COMMAND ----------


features_to_remove = {'Id'}

approved_cols = [col for col in df.columns if (col not in features_to_remove)]

stats_options = tfdv.StatsOptions(feature_allowlist=approved_cols)
print(stats_options.feature_allowlist)


# COMMAND ----------

train_stats = tfdv.generate_statistics_from_dataframe(train_df, stats_options)

# COMMAND ----------


print(f"Number of features used: {len(train_stats.datasets[0].features)}")

print(f"Number of examples used: {train_stats.datasets[0].num_examples}")

print(f"First feature: {train_stats.datasets[0].features[0].path.step[0]}")
print(f"Last feature: {train_stats.datasets[0].features[-1].path.step[0]}")

# COMMAND ----------

tfdv.visualize_statistics(train_stats)

# COMMAND ----------

schema = tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema)

# COMMAND ----------


print(f"Number of features in schema: {len(schema.feature)}")

print(f"Second feature in schema: {list(schema.feature)[1].domain}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Validation Step 4: Evaluation

# COMMAND ----------

eval_stats = tfdv.generate_statistics_from_dataframe(eval_df, stats_options=stats_options)
tfdv.visualize_statistics(lhs_statistics=eval_stats, rhs_statistics=train_stats,
                          lhs_name='EVAL_DATASET', rhs_name='TRAIN_DATASET')

# COMMAND ----------


print(f"Number of features: {len(eval_stats.datasets[0].features)}")

print(f"Number of examples: {eval_stats.datasets[0].num_examples}")

print(f"First feature: {eval_stats.datasets[0].features[0].path.step[0]}")
print(f"Last feature: {eval_stats.datasets[0].features[-1].path.step[0]}")

# COMMAND ----------

def calculate_and_display_anomalies(statistics, schema):
    '''
    Calculate and display anomalies.

            Parameters:
                    statistics : Data statistics in statistics_pb2.DatasetFeatureStatisticsList format
                    schema : Data schema in schema_pb2.Schema format

            Returns:
                    display of calculated anomalies
    '''
    ### START CODE HERE
    # HINTS: Pass the statistics and schema parameters into the validation function 
    anomalies = tfdv.validate_statistics(statistics, schema)
    
    # HINTS: Display input anomalies by using the calculated anomalies
    tfdv.display_anomalies(anomalies)
    ### END CODE HERE

# COMMAND ----------

calculate_and_display_anomalies(eval_stats, schema=schema)

# COMMAND ----------

# MAGIC %md
# MAGIC # Validation Step 5: Serving

# COMMAND ----------

options = tfdv.StatsOptions(schema=schema, 
                            infer_type_from_schema=True, 
                            feature_allowlist=approved_cols)

# COMMAND ----------

schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')

# COMMAND ----------

serving_stats = tfdv.generate_statistics_from_dataframe(serving_df, stats_options=options)
tfdv.get_feature(schema, 'Response').not_in_environment.append('SERVING')
serving_anomalies_with_env = tfdv.validate_statistics(serving_stats, schema, environment='SERVING')
tfdv.display_anomalies(serving_anomalies_with_env)

# COMMAND ----------

skew_drift_anomalies = tfdv.validate_statistics(train_stats, schema,
                                          previous_statistics=eval_stats,
                                          serving_statistics=serving_stats)

# COMMAND ----------

tfdv.display_anomalies(skew_drift_anomalies)

# COMMAND ----------

tfdv.write_schema_text(schema, 'schema.pbtxt')  

# COMMAND ----------

train_df.to_csv('train_df.csv')
eval_df.to_csv('eval_df.csv')
serving_df.to_csv('serving_df.csv')