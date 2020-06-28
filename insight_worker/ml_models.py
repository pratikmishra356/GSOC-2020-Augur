
import pandas as pd
import requests
import datetime
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from numpy import hstack
import seaborn as sns
import random
from functools import reduce
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,Activation
from keras import optimizers
from keras.layers import Dense,Dropout
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import STL
from sklearn.cluster import AgglomerativeClustering,KMeans

import logging

def cluster_model(self,entry_info,repo_id,df):
    
    data = pd.read_csv("repo_reviews_all.csv")
    data.index = data.iloc[:,0]
    data.drop(['repo_id'],axis=1,inplace=True)
    dfs = []
    dfs.append(data)
    dfs.append(pd.DataFrame(df.iloc[:,1]).T)
    data = pd.concat(dfs)
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data))
    data_scaled.index = data.index

    
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward').fit_predict(data_scaled)
    x = cluster.labels_
    unique, counts = np.unique(x, return_counts=True)
    
    df_frame = pd.DataFrame(data.sum(axis=1))
    df_frame['cluster'] = x
    df_frame.columns = ['metric','cluster']
    
    clusters_means={}
    for k in unique:
        filt = df_frame['cluster'] == k
        df_frame[filt]['metric'].mean()
        
        clusters_means[k] = df_frame[filt]['metric'].mean()
        
    clusters_means = {ke: v for ke, v in sorted(clusters_means.items(), key=lambda item: item[1])}
    xp = clusters_means
    df_scaled = scaler.transform(df.iloc[:,1].T)
    
    pred = x[-1]
    
    if pred == xp[0]:
        #return 'less_active'
        stl_less_active(self,entry_info,repo_id,df)
    elif pred == xp[1]:
        #return 'moderate_active'
        lstm_moderate_active(self,entry_info,repo_id,df)
    else:
        #return 'highly_active'
        lstm_highly_active(self,entry_info,repo_id,df)


    


def preprocess_data(data,tr_days,lback_days,n_features,n_predays):
    
    train_data = data.values

    features_set = []
    labels = []
    for i in range(lback_days, tr_days+1):
        features_set.append(train_data[i-lback_days:i,0])
        labels.append(train_data[i:i+n_predays, 0])

    features_set = np.array(features_set)
    labels = np.array(labels)

    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], n_features))

    
    return features_set,labels


def model_lstm(features_set,n_predays,n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(90, activation='linear',return_sequences=True, input_shape=(features_set.shape[1], n_features))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(90, activation='linear',return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(90, activation='linear')))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='adam', loss='mae')
    return model



def stl_less_active(self,entry_info,repo_id,df_less_active):
    
    stl = STL(df_less_active.iloc[:,1], seasonal=5,trend=47,period=45,robust=True)
    res = stl.fit()
    
    std_resid = np.std(res.resid)
    mean_resid = np.mean(res.resid)
    
    filt = (res.resid > 3*s) | (res.resid < -3*s)
    
    anomaly_df = df_less_active[filt]
    anomaly_df['score'] = res.resid[filt]  ##### date, metric_name(will contain real value) , score(anomaly_score)

    
    ### Inserting anomaly in database

    anomaly_df_copy = anomaly_df.copy()

    insight_count = 0
    next_recent_anomaly_date = anomaly_df.idxmax()
    logging.info("Next most recent date: \n{}\n".format(next_recent_anomaly_date))
    next_recent_anomaly = anomaly_df.loc[anomaly_df.index == next_recent_anomaly_date]
    logging.info("Next most recent anomaly: \n{}\n{}\n".format(next_recent_anomaly.columns.values, 
            next_recent_anomaly.values))

    if insight_count == 0:
        most_recent_anomaly_date = next_recent_anomaly_date
        most_recent_anomaly = next_recent_anomaly

    split = df_highly_active.columns.split(" - ")

    for tuple in anomaly_df_copy.itertuples():
        try:
            # Format numpy 64 date into timestamp
            date64 = tuple.Index
            ts = (date64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            ts = datetime.datetime.utcfromtimestamp(ts)

            data_point = {
                'repo_id': repo_id,
                'ri_metric': split[0],
                'ri_field': split[1],
                'ri_value': tuple._3,
                'ri_date': ts,
                'ri_fresh': 0 if date64 < most_recent_anomaly_date else 1,
                'ri_score': most_recent_anomaly.iloc[0]['score'],
                'ri_detection_method': 'time_series_LSTM_model',
                "tool_source": self.tool_source,
                "tool_version": self.tool_version,
                "data_source": self.data_source
            }
            result = self.db.execute(self.repo_insights_table.insert().values(data_point))
            logging.info("Primary key inserted into the repo_insights table: {}\n".format(
                result.inserted_primary_key))

            logging.info("Inserted data point for metric: {}, date: {}, value: {}\n".format(self.metric, ts, tuple._3))
        except Exception as e:
            logging.info("error occurred while storing datapoint: {}\n".format(repr(e)))
            break
    

def lstm_moderate_active(self,entry_info,repo_id,df_moderate_active):
    
    
    scaler = MinMaxScaler(feature_range=(0,1))

    data = pd.DataFrame(df_moderate_active.iloc[:,1])
    data = pd.DataFrame(scaler.fit_transform(data.values))

    tr_days = 351
    lback_days = 14
    n_features = 1
    n_predays = 1

    features_set,labels = preprocess_data(data,tr_days,lback_days,n_features,n_predays)
    model = model_lstm(features_set,n_predays,n_features)

    history = model.fit(features_set, labels, epochs = 50, batch_size = 10,validation_split=0.1,verbose=0).history
     
    
    test_inputs = data[ :len(df_moderate_active.iloc[:,1])].values
    test_inputs = test_inputs.reshape(-1,n_features)
    test_features = []
    for i in range(lback_days, len(df_moderate_active.iloc[:,1])):
        test_features.append(test_inputs[i-lback_days:i, 0])

    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], n_features))
    predictions = model.predict(test_features)
    predictions = scaler.inverse_transform(predictions)
    
    
    #Finding anomalies
    test_data = df_moderate_active.iloc[14:,1]
    error = np.array(test_data[:]- predictions[:,0])
    std_error = np.std(error)
    mean_error = np.mean(error)
    
    filt = ( error > 2.5*std_error) | ( error < - 2.5*std_error) 
    anomaly_df = df_moderate_active.iloc[14:][filt]
    anomaly_df['score'] = error[filt]   ##### date, metric_name(will contain real value) , score(anomaly_score)
    
    
    ### Inserting anomaly in database
    anomaly_df_copy = anomaly_df.copy()

    insight_count = 0
    next_recent_anomaly_date = anomaly_df.idxmax()
    logging.info("Next most recent date: \n{}\n".format(next_recent_anomaly_date))
    next_recent_anomaly = anomaly_df.loc[anomaly_df.index == next_recent_anomaly_date]
    logging.info("Next most recent anomaly: \n{}\n{}\n".format(next_recent_anomaly.columns.values, 
            next_recent_anomaly.values))

    if insight_count == 0:
        most_recent_anomaly_date = next_recent_anomaly_date
        most_recent_anomaly = next_recent_anomaly

    split = df_moderate_active.columns.split(" - ")

    for tuple in anomaly_df_copy.itertuples():
        try:
            # Format numpy 64 date into timestamp
            date64 = tuple.Index
            ts = (date64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            ts = datetime.datetime.utcfromtimestamp(ts)

            data_point = {
                'repo_id': repo_id,
                'ri_metric': split[0],
                'ri_field': split[1],
                'ri_value': tuple._3,
                'ri_date': ts,
                'ri_fresh': 0 if date64 < most_recent_anomaly_date else 1,
                'ri_score': most_recent_anomaly.iloc[0]['score'],
                'ri_detection_method': 'time_series_LSTM_model',
                "tool_source": self.tool_source,
                "tool_version": self.tool_version,
                "data_source": self.data_source
            }
            result = self.db.execute(self.repo_insights_table.insert().values(data_point))
            logging.info("Primary key inserted into the repo_insights table: {}\n".format(
                result.inserted_primary_key))

            logging.info("Inserted data point for metric: {}, date: {}, value: {}\n".format(self.metric, ts, tuple._3))
        except Exception as e:
            logging.info("error occurred while storing datapoint: {}\n".format(repr(e)))
            break
    

def lstm_highly_active(self,entry_info,repo_id,df_highly_active):
    
    scaler = MinMaxScaler(feature_range=(0,1))

    data = pd.DataFrame(df_highly_active.iloc[:,1])
    data = pd.DataFrame(scaler.fit_transform(data.values))

    tr_days = 351
    lback_days = 30
    n_features = 1
    n_predays = 1

    features_set,labels = preprocess_data(data,tr_days,lback_days,n_features,n_predays)
    model = model_lstm(features_set,n_predays,n_features)

    history = model.fit(features_set, labels, epochs = 50, batch_size = 30,validation_split=0.1,verbose=0).history
    
    
    
    test_inputs = data[ :len(df_highly_active.iloc[:,1])].values
    test_inputs = test_inputs.reshape(-1,n_features)
    test_features = []
    for i in range(lback_days, len(df_highly_active.iloc[:,1])):
        test_features.append(test_inputs[i-lback_days:i, 0])

    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], n_features))
    predictions = model.predict(test_features)
    predictions = scaler.inverse_transform(predictions)
    
    
    #Finding anomalies
    test_data = df_highly_active.iloc[30:,1]
    error = np.array(test_data[:]- predictions[:,0])
    std_error = np.std(error)
    mean_error = np.mean(error)
    
    filt = ( error > 2*std_error) | ( error < - 2*std_error) 
    
    anomaly_df = df_highly_active.iloc[30:][filt]
    anomaly_df['score'] = error[filt] ##### date, metric_name(will contain real value) , score(anomaly_score)
    
    
    ### Inserting anomaly in database
    
    anomaly_df_copy = anomaly_df.copy()

    insight_count = 0
    next_recent_anomaly_date = anomaly_df.idxmax()
    logging.info("Next most recent date: \n{}\n".format(next_recent_anomaly_date))
    next_recent_anomaly = anomaly_df.loc[anomaly_df.index == next_recent_anomaly_date]
    logging.info("Next most recent anomaly: \n{}\n{}\n".format(next_recent_anomaly.columns.values, 
            next_recent_anomaly.values))

    if insight_count == 0:
        most_recent_anomaly_date = next_recent_anomaly_date
        most_recent_anomaly = next_recent_anomaly

    split = df_highly_active.columns.split(" - ")

    for tuple in anomaly_df_copy.itertuples():
        try:
            # Format numpy 64 date into timestamp
            date64 = tuple.Index
            ts = (date64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            ts = datetime.datetime.utcfromtimestamp(ts)

            data_point = {
                'repo_id': repo_id,
                'ri_metric': split[0],
                'ri_field': split[1],
                'ri_value': tuple._3,
                'ri_date': ts,
                'ri_fresh': 0 if date64 < most_recent_anomaly_date else 1,
                'ri_score': most_recent_anomaly.iloc[0]['score'],
                'ri_detection_method': 'time_series_LSTM_model',
                "tool_source": self.tool_source,
                "tool_version": self.tool_version,
                "data_source": self.data_source
            }
            result = self.db.execute(self.repo_insights_table.insert().values(data_point))
            logging.info("Primary key inserted into the repo_insights table: {}\n".format(
                result.inserted_primary_key))

            logging.info("Inserted data point for metric: {}, date: {}, value: {}\n".format(self.metric, ts, tuple._3))
        except Exception as e:
            logging.info("error occurred while storing datapoint: {}\n".format(repr(e)))
            break
 

def time_series_LSTM_model(self,entry_info,repo_id,df):


    for i in range(1,df.columns.shape[0]):
        
        cluster_model(self,entry_info,repo_id,df.iloc[:,[0,i]])