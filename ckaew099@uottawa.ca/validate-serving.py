# Databricks notebook source
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
import warnings


# Set TF's logger to only display errors to avoid internal warnings being shown
tf.get_logger().setLevel('ERROR')

# COMMAND ----------

schema = tfdv.load_schema_text(
    'schema/schema.pbtxt'
) 

# COMMAND ----------

serving_df = spark.read.table('`customer`')
serving_df = serving_df.toPandas()
serving_df['Dt_Customer'] = pd.to_datetime(serving_df['Dt_Customer'], errors='coerce')
serving_df['Dt_Customer'] = serving_df['Dt_Customer'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)

# COMMAND ----------


features_to_remove = {'Id'}

approved_cols = [col for col in serving_df.columns if (col not in features_to_remove)]

stats_options = tfdv.StatsOptions(feature_allowlist=approved_cols)
print(stats_options.feature_allowlist)

# COMMAND ----------

options = tfdv.StatsOptions(schema=schema, 
                            infer_type_from_schema=True, 
                            feature_allowlist=approved_cols)

# COMMAND ----------

serving_stats = tfdv.generate_statistics_from_dataframe(serving_df, stats_options=options)
tfdv.get_feature(schema, 'Response').not_in_environment.append('SERVING')
serving_anomalies_with_env = tfdv.validate_statistics(serving_stats, schema, environment='SERVING')
tfdv.display_anomalies(serving_anomalies_with_env)

# COMMAND ----------

if serving_anomalies_with_env.anomaly_info != {}:
    warnings.warn(f'TFDV detect anomalies in serving data, please be careful {serving_anomalies_with_env.anomaly_info}', UserWarning)
