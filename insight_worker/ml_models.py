
import pandas as pd
import requests
import datetime
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from numpy import hstack
import seaborn as sns
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense,Dropout
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

import logging



def time_series_LSTM_model(self,entry_info,repo_id,df):

    def preprocess_data(data,tr_days,lback_days,n_features,n_predays):
    
        train_data = data.values

        features_set = []
        labels = []
        for i in range(lback_days, tr_days+1):
            features_set.append(train_data[i-lback_days:i, 0])
            labels.append(train_data[i:i+n_predays, 0])

        features_set = np.array(features_set)
        labels = np.array(labels)

        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], n_features))
        
        
        return features_set,labels

    def model_lstm(fetures_set,n_predays,n_features):
        model = Sequential()
        model.add(LSTM(60, activation='linear', return_sequences=True, input_shape=(features_set.shape[1], n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(40, activation='linear',return_sequences=False))
        model.add(Dense(n_predays))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model

    scaler = MinMaxScaler()

    data = pd.DataFrame(df.iloc[:,0])
    data = pd.DataFrame(scaler.fit_transform(data.values))

    tr_days = 351
    lback_days = 3
    n_features = 1
    n_predays = 1

    features_set,labels = preprocess_data(data,tr_days,lback_days,n_features,n_predays)
    model = model_lstm(features_set,n_predays,n_features)

    history = model.fit(features_set, labels, epochs = 80, batch_size = 5,validation_split=0.1,verbose=0).history



    test_inputs = data[tr_days-lback_days :len(df)].values
    test_inputs = test_inputs.reshape(-1,n_features)
    test_features = []
    for i in range(lback_days, len(df)-tr_days+lback_days):
        test_features.append(test_inputs[i-lback_days:i, 0])
        
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], n_features))
    predictions = model.predict(test_features)

    predictions = scaler.inverse_transform(predictions)

    actual_values = df.iloc[tr_days:len(df),0].values
    

    score_train = pd.DataFrame(index=df[tr_days:].index)
    score_train['Loss_mae'] = np.abs(predictions-actual_values)
    score_train['Threshold'] = 0.15
    score_train['Anomaly'] = score_train['Loss_mae'] > score_train['Threshold']
    anomaly_df = df.loc[score_train['Anomaly'] == True, df.columns]
    anomaly_df['score'] = score_train['Loss_mae']
   
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

    split = df.columns.split(" - ")

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

