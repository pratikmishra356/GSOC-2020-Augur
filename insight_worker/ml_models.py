
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
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

import logging



def time_series_LSTM_model(self,entry_info,repo_id,df):

    columns = df.columns
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df))
    df.columns = columns



    def lstm_model(data,days):
    
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(days, data.shape[1])))
        model.add(LSTM(52, activation='relu',return_sequences=True))
        model.add(Dense(data.values.shape[1]))
        model.compile(optimizer='adam', loss='mae')
        return model


    days = 1
    nb_epochs = 50

    batch_size = 5

    data = df.iloc[0:(int(df.shape[0]/days)*days)]

    X_train = data.values.reshape(int(data.values.shape[0]/days),days, data.shape[1])
    model = lstm_model(data,days)
    model.fit(X_train,X_train,epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.08,verbose=0)
    
    X_predict = model.predict(X_train)
    X_predict = X_predict.reshape(X_predict.shape[0]*days, X_predict.shape[2])
    X_predict = pd.DataFrame(X_predict, columns=df.columns)
    X_predict.index = df.index

    Xtrain = X_train.reshape(X_train.shape[0]*days, X_train.shape[2])


    score_train = pd.DataFrame(index=df.index)
    score_train['Loss_mae'] = np.mean(np.abs(X_predict-Xtrain), axis = 1)
    score_train['Threshold'] = 0.006
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





def next_ml_model(self,df):
    pass