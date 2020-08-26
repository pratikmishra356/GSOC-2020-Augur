# Augur-GSoC-2020 Project Summary

## Machine Learning for Anomaly Detection in Open Source Communities

Augur is an open source platform that systematically integrates data from several open source repositories, issue trackers, mailing lists, and other communication systems that open source projects rely on to create a highly structured (relational and graph databases), consistent, and validated collection of open source health and sustainability data. Hundreds of highly specialized data requests are implemented in Augur's API, data and visualizations are pushed to Augur users, and the results of one user request benefits the whole community.


## My Contributions 

### Implemented Insight_Worker for time_series metrics [pr864](https://github.com/chaoss/augur/pull/864) .
	
Implemented  
	
* time_series_metrics function for data collection and initial data preprocessing
* model_lstm method for LSTM model configuration
* preprocess_data method for arranging data for lstm_keras method
* lstm_selection method for selection of time_step parameters for different repo fields
* lstm_keras method for training and predicting outliers and insertion of new models and summary into corresponding tables
* insert_data method for insertion of outliers into repo_insights and repo_insights_records table
* Notebooks for Data analysis and different lstm_model results evaluations is present in time_series_notebook

### Testing, Documentation and final reference Notebook [pr890](https://github.com/chaoss/augur/pull/890)

* Added Testing and Documentation
* Added Reference notebook for better understanding of model implemented

	
## Project Summary

### Problem Statement:

The volume of activity across all dimensions of open source makes the identification of significant changes both labor intensive and impractical. By connecting Augur's "insight worker" to its "push notification" architecture, and related pages that allow exploration of identified anomalies, open source companies, community managers, and contributors will be in a better position to identify community or technology issues quickly.

We need to implement a Machine Learning model inside the insight_worker to discover insights and outliers in the time_series metrics of over 100,000 repositories.Model should be able to discover outliers on a wide variety of data over more than 20 time_series metrics.It should also be user friendly, so that user can control the outcome according to their need.

Data can be fetched using the Augur’s APIs which contains endpoints of several metrics which were already defined.Metrics that will be analysed in this worker would be mainly time_series metrics like commit_counts, code_changes_lines, issues_new, issues_active, issues_closed, reviews, reviews_accepted, reviews_declined, new_contributors etc.


### Challenges :

So, we have multi-dimensional data with multiple sources,that is each repository will have a sequence of data from multiple above mentioned metrics.Also these repos will have different characteristics like their level of activity either high,low or moderate and scale of their workflow such as ranges of values in the time_series_metrics.
We can see that data is diverse and we need to select such models which can handle these types of data with automatic tuning of parameters according to the data.


### Possible Solutions :

There are many algorithms that were useful for outlier detection in time_series data like **ARIMA, SARIMA, STL decomposition, IsolationForest, KNN, SVM, CAD OSE, Skyline, Prophet(by facebook), Numenta HTM , LSTM,** etc.

* **ARIMA** : Autoregressive Integrated Moving Average, or ARIMA, is one of the most widely used forecasting methods for univariate time series data forecasting. Any ‘non-seasonal’ time series that exhibits patterns and is not a random white noise can be modeled with ARIMA models.A standard notation is used of ARIMA(p,d,q) where the parameters are substituted with integer values to quickly indicate the specific ARIMA model being used.\
\
Problem with this model is that it does not support seasonal data,so data with the repeating cycle needs to be adjusted with seasonal differencing.It accepts three parameters p,d,q.Selecting the optimal values for p,d,q needs inspection of data with ACF and PACF plots which can not  be feasible for such large number of repos.




* **SARIMA** :  Seasonal Autoregressive Integrated Moving Average, or SARIMA, method for time series forecasting with univariate data containing trends and seasonality.As an extension of the ARIMA method, the SARIMA model not only captures regular difference, autoregressive, and moving average components as the ARIMA model does but also handles seasonal behavior of the time series. In the SARIMA model, both seasonal and regular differences are performed to achieve stationarity prior to the fit of the ARMA model.\
\
Problem with this model is that data needs to be stationary.A time series is said to be stationary if its statistical properties such as mean, variance and covariance remain constant over time.Now it's hard to find the seasonal and trend parameters as the data has a wide range of diversity.

* **STL Decomposition** :STL stands for "Seasonal and Trend decomposition using Loess" and splits time series into trend, seasonal and remainder components. Advantages of the STL decomposition are that the parameter robust can be set to true to remove effects of outliers on the calculation of trend and seasonal component.\
\
To perform the STL decomposition, the seasonality of the data has to be known.It's hard to find the seasonality in every repos as different repos might have different patterns and scale of workflow.Also the parameters used in the STL is sensitive which highly effects the residual value.


* **IsolationForest** : The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.\
\
The main problem with the  algorithm is that the way the branching of trees takes place introduces a bias, which is likely to reduce the reliability of the anomaly scores for ranking the data.Also it lacks in finding the sequences in the data and focuses more on extreme values which is not sufficient for time series data.


* **CAD-OSE / KNN-CAD / SVM-CAD** : [paper](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-014-0011-y) , [example](https://github.com/smirmik/CAD)\
Contextual anomaly detection seeks to find relationships within datasets where variations in external behavioural attributes well describe anomalous results in the data. \
\
Contextual anomaly applications are normally handled in one of two ways.
First, the context anomaly problem is transformed into a point anomaly problem. That is, the application attempts to apply separate point anomaly detection techniques to the same dataset, within different contexts.\
\
The second approach to handling contextual anomaly applications is to utilize the existing structure within the records to detect anomalies using all the data concurrently.


* **SKYLINE** : [link](https://github.com/earthgecko/skyline) Skyline is a Python based anomaly detection/deflection stack that analyses, anomaly detects, deflects, fingerprints and learns vast amounts of streamed time series data.
Skyline is a near real time anomaly detection system, built to enable passive monitoring of hundreds of thousands of metrics, without the need to configure a model/thresholds for each one.\
\
Skyline implements a novel time series similarities comparison algorithm and a boundary layers methodology that generates fingerprints of time series data using the sum of the values of features of the time series.


* **PROPHET** : [link](https://facebook.github.io/prophet/)
Prophet is an open source library published by Facebook that is based on decomposable (trend+seasonality+holidays) models. Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.\
\
The Prophet procedure includes many possibilities for users to tweak and adjust forecasts. You can use human-interpretable parameters to improve your forecast by adding your domain knowledge.







* **NUMENTA HTM** : [link](https://numenta.com/blog/2019/10/24/machine-learning-guide-to-htm)\
The Hierarchical Temporal Memory(HTM) algorithm is based on the well understood principles and core building blocks of the Thousand Brains Theory. In particular, it focuses on three main properties: *sequence learning, continual learning and sparse distributed representations*.\
\
HTM sequence memory not only advances our understanding of how the brain may solve the sequence learning problem, but is also applicable to a wide range of real-world problems such as discrete and continuous sequence prediction, anomaly detection, and sequence classification.\
\
HTM builds sparse, invariant representations of pattern sequences representing repeated structures in the input stream. The algorithm learns which patterns are likely to follow each other, thus learning to predict future patterns. 


* **LSTM** : [link](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies.
LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn.\
\
The core concept of LSTM’s are the cell state, and it’s various gates. The cell state acts as a transport highway that transfers relative information all the way down the sequence chain.The LSTM  does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.



## My Solution : 

After analysing all the above mentioned algorithms we tried to implement a simple yet effective model which should possess most of the benefits present in the above mentioned algorithms.The Numenta Anomaly Benchmark(NAB) is a novel benchmark for evaluating algorithms for anomaly detection in streaming, real-time applications. It is composed of over 50 labeled real-world and artificial time series data files plus a novel scoring mechanism designed for real-time applications.In this repository they ranked some of the most effective algorithms with their performance values.

I analysed these algorithms,their architecture, application, performance and then tried to implement those ideas into our model so that we can customise whenever we need to do so.

* ***Data Analysis*** :  
I also did detailed analysis of the different metrics for different repositories.In the analysis found that the repositories can be divided into three separate categories based on their level of activity like highly,moderate and less active repos.Also different metrics have different range of values like code-changes-lines metrics has much higher range of values than the issues, pull_requests, commt_counts, new_contributors etc.So, I decided to cluster the repos based on their activity level considering the metrics individually.\
\
After plotting multiple dendrograms and clustering model we came up with three different models DBSCAN, KMeans, Agglomerative clustering.But implementation of these models and predicting class of each repos need huge amount of fetching data from the database which is not feasible and so decided to somehow automate these classification method.



* ***BiLSTM implementation*** :
A long short-term memory network(LSTM) is one of the most commonly used neural networks for time series analysis.The ability of LSTM to remember previous information makes it ideal for such tasks.\
\
There is no doubt that LSTM is good in detecting patterns but the result of it mostly depends on your data.Data visualisation is the key feature in getting desired results.\
\
So, the basic idea was to predict the next time_step value based on some set of previous time_step values and after that finding the difference between the actual and predicted value.If the difference is too large then consider them as outliers.\
\
**LSTM** takes two parameters first the number of neurons and second input shape.Now input shape of LSTM must be three dimensional consisting of number of samples , time_steps , features.Also if you may mention the return_sequence and activation function as parameters.\
\
Let's say I have data from 1st Jan to 31 Dec and I want to predict the value of 1st Jan next year ,so I will use these 365 days data for training.We will transform this unsupervised learning into supervised learning by giving values of some days in past as X and the next day as Y.\
\
***samples*** : It is the number of instances you use for training.Like if we use 365 days of training with time_step = 7 , then samples will be equal to 358 and your last input instance will be x =[358,359,360,361,362,363,364],y=[365] if you want to predict one day or step in future.And for predicting after 365 days then your test input instance should be x = [359,360,361,362,363,364,365].But let's say using past 7 days record you want to predict two days values in future , then your last instance would be x =[357,358,359,360,361,362,363],y=[364,365] and total instances would be 357.\
\
***time_step*** : So, number of days we will pass as X for one instance is time_step.Now this one instance of X and Y values is one sample.If we have 365 days then we will use chunks of past week as X and the very next day as Y if we want to predict one day or step in future.\
\
***features*** : This is the number of different variables you are using for training purpose.Lets say your data has features [“closed”,”open” ,”active”,”inactive”].Now you want to predict ‘closed’ variable using all the features as input including ‘closed’ then features =4, if excluding ‘closed’ as input then features = 3.\
\
Now let's talk about output shape, in the LSTM model at last we add a dense layer with a certain number of neurons , that number is decided by how many days or steps we want to predict in future.\
\
I tried different LSTM models like Vanilla LSTM(single lstm hidden layer), Stacked LSTM(multiple hidden LSTM layers stacked on top of one another), Bidirectional LSTM(it allows the LSTM layer to learn the input sequence both backward and forward), CNN LSTM(convolutional neural network LSTM layer) with different hyperparameters,  optimisers, activation functions.\
\
After reading lots of articles , applying several LSTM models over different types of data , I understood in which fields and how it will perform better and so decided to use Bidirectional LSTM due to the following reasons:

* **In problems where all timesteps of the input sequence are available, Bidirectional LSTMs train two instead of one LSTMs on the input sequence. The first on the input sequence as-is and the second on a reversed copy of the input sequence.**

* **It differs from unidirectional LSTM as it has two networks  that run both backward and forward thus it can preserve the information from both past and future.**

* **BiLSTMs show very good results as they can understand context better due to its Bidirectional networks.**
\
\
Now let's discuss the implementation of different methods present in this model.\
\
***time_series_metrics*** : In this method endpoints of the time_series metrics are being called and data is fetched using Augur’s APIs.Then after fetching this data,we create a dataframe with the first column as “date” and other columns as metrics fields.This dataframe is structured in such way that it can be used for any Machine Learning model.\
\
***lstm_selection*** : As discussed earlier, repos have different levels of activity and different ranges of values.Time_step parameter in the BiLSTM model is the key to model performance as model summarise all the neurons values for that time_step and then predict the next time_step.In case of low time_step like 3 to 7 model output is highly correlated with its past 2,3 days values,However in large time_step like 20 to 30 model output is not correlated with any particular 2,3 days values, instead its prediction depends on the seasonality,trend over this large time_step period.\
\
Now,large time_step values are useful for the repos which have a high level of activity,where there is continuous distribution of values.While low time_step are useful for the repos which have low level of activity as for this repos the last few days value is enough to predict the next day value.\
\
So, we defined an equation which can calculate the time_step value based on the statistical analysis of the data.Level of activity of the metrics can be calculated by checking the sparsity in the data , high sparsity means low activity while range of values can be calculated by coefficient of variation in the data,higher variation means lower ranges of values.Coefficient of variation can be affected by the outliers so, we consider only the 90 percentile of the data to calculate C.V.\
\
Just like SKYLINE this method enables the model to handle large amounts of metrics explicitly.This also removes the problem of fetching huge amounts of data for clustering of repos.\
\
***model_lstm*** : It contains the configuration of the BiLSTM model.It contains three Bidirectional LSTM layers stacked on top of one another with 90 neurons in each layer.As the model will predict only the values so,the output will be continuous and not discrete thats why “linear” activation function has been chosen.\
\
There is certain limitations with other popular activation functions “ReLu”,”sigmoid”,”tanh” etc.A general problem with both the sigmoid and tanh functions is that they saturate.The functions are only really sensitive to changes around their mid-point of their input, such as 0.5 for sigmoid and 0.0 for tanh.Llimitations of ReLU is when large weight updates can mean that the summed input to the activation function is always negative, regardless of the input to the network.\
\
Also the output layer consists of a Dense layer with one neuron to output the very next time_step value which is sufficient according to our problem statement,but this can be changed as per our need.\
\
This configuration of the model is designed to incorporate robust features of the HTM algorithm by using stacked BiLSTM.\
\
**1**.HTMs are particularly suited for sequence learning modeling as the latest RNNs incarnation such as LSTMs or GRUs.Although HTMs algorithmic approach and learning rule differ at some level.\
\
**2**.HTM requires explicitly timestamped data to recognize daily and weekly patterns, while this model only needs the raw sequential data to predict such time series accurately.\
\
***preprocess_data*** : This method arranges training_data according to different parameters passed by lstm_keras method for the BiLSTM model.Currently we are using univariate variable method i.e we predict values in metric based on its previous values without considering its dependency on other metrics ,that's why the feature value is one.\
\
However, we can implement the idea of contextual anomaly detection(CAD) by evaluating the correlation between the metrics.For example, number of issues closed on any particular day will not only depend on issues closed in the last instance of time_step but will also depend on the number of issues open and number of issues active in the last instance of time_steps.So, instead of adding features value equal to 1, we can add multiple feature variable to improve the prediction.\
\
**lstm_keras** : In this method fitting of the model and prediction of value takes place.We already gone through various methods that will be implemented inside this method.After predicting values outliers discovered based on the Rolling Mean and Rolling Standard deviation of the error(actual values - prediction values).\
\
Model predicts the value for the whole training days in order to analyse the model performance for the whole sequence which we store in the table for further model performance analysis.As our goal is to find the outliers, so if we consider the mean and standard deviation(std) for the whole sequence, then the global or large outliers might affect these values and it is possible that we miss the local outliers.In order to avoid this, we first discover the global outliers and then calculates the mean and std for each instance of time_step.If the error in the prediction of next time_step exceeds then combination of mean and std of the last instance of time_step, then we consider this as local outliers.


After considering key features of the top ranked models like SKYLINE,CAD,NUMENTA HTM etc. that were evaluated on 50 labeled real-world and artificial time series data files and were ranked by novel scoring mechanism designed for real-time applications,
this whole setup of the model designed to give robust performance in detecting outliers irrespective of the nature of the repos and their metrics.



## Result Analysis :

The BiLSTM model tested on a set of 15 repositories with multiple metrics like code-changes-lines, issues, pull_requests etc.

* Model performed pretty well and was able to detect trends and seasonality in the metrics.

* As this is an unsupervised machine learning problem,So data were labeled manually and then precision and recall were calculated and were found above 0.8 which is quite impressive.

* Also the model performance analysed based on its prediction values over the training period and ratio of error upon actual values were below 0.5 in most of the cases.

* However model struggles in code-changes-lines metrics due to the wide range of values and irregular distribution of the data.These can be solved by logarithmic transformation only for these metrics.

* Parameters in the BiLSTM model automatically changes according to the data which makes it suitable for almost all types of data.




## Further improvements : 


Machine Learning is all about improving time by time and so certain improvements or new approaches can be implemented.

* Model struggles in code-changes-lines metrics, so this can be improved by implementing certain extra data transformation methods to handle this metric.

* Certain feature engineering can be implemented like finding the dependency  of metrics on one another and to evaluate if this can improve models performance further.


## Project Development 



## Community Bonding - May 5th to May 31st, 2020

* Blog : [GSoC Community Bonding](https://medium.com/@pratikmishra_60029/expected-outcomes-discussed-during-gsoc-community-bonding-period-1-84cee2d3f277)


 ## Week 1


* Implemented Insight_worker structure
* Implemented preprocess_metric method
* Implemented stacked LSTM model
* [Notebook - LSTM](https://github.com/pratikmishra356/augur_ML_insights/blob/master/insight_model_LSTM.ipynb)
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-1-2-2-7f3c52d26f07)

## Week 2


* Categorised over 3000 repos
* Exploratory data analysis on each repo category
* Implemented STL_decompostiotn
* [Notebook - Data analysis](https://github.com/pratikmishra356/augur_ML_insights/blob/master/data_analysis.ipynb)
* [Notebook - STL_decomposition](https://github.com/pratikmishra356/augur_ML_insights/blob/master/STL_decomposition.ipynb)
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-1-2-2-7f3c52d26f07)


## Week 3


* Clustered 3000 repos with different time series metrics using Unsupervised clustering algorithms
* Plotted dendrogram to get an idea of number of clusters using euclidean and pearson correlation distance
* Implemented three clustering alogrithm DBSCAN ,KMeans and AgglomerativeClustering( hierarchical clustering)
* [Notebook - repo_new-issues_cluster](https://github.com/pratikmishra356/augur_ML_insights/blob/master/Notebooks/repo_new-issues_cluster.ipynb)
* [Notebook - repo_reviews_cluster](https://github.com/pratikmishra356/augur_ML_insights/blob/master/Notebooks/repo_reviews_cluster.ipynb)
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-3-4-3-edd6952a129d)

## Week 4


* Implemented BiDirectional Stacked LSTM model on moderate and highly active repos
* Implemented STL decompostion on less active repos
* Combine clustering and outlier detection models and created a pipeline
* [Notebook - Insight_new-issues](https://github.com/pratikmishra356/augur_ML_insights/blob/master/Notebooks/Insight_new-issues.ipynb)
* [Notebook - Insight_reviews](https://github.com/pratikmishra356/augur_ML_insights/blob/master/Notebooks/Insight_reviews.ipynb)
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-3-4-3-edd6952a129d)


## Week 5


* Implementation of ML pipeline into the insight_worker
* Integrated different methods like time_series_metrics,clustering_algorithm into the insight_worker
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-5-6-4-88d5cc179cb4)


## Week 6


* Single time training of clustering model and automated clustering of repos by pickeling the model.
* Detection of both local and global outliers by moving the threshold value
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-5-6-4-88d5cc179cb4)


## Week 7


* Performed two model with different time steps on categorically separated repos.
* Analysis of performance of model based on contamination_factor and ratio of actual values upon error
* [Notebook - Outliers results using LSTM models with different time_steps](https://github.com/chaoss/augur/blob/pratik/time_series_notebook/Distribution%20and%20model%20results%20analysis%20on%20individual%20data_fields.ipynb)
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-7-8-5-c450f35cf46e)


## Week 8


* Automated the selection of time_steps for the model based on sparsity and co-efficient of variation in data.
* Implemented method for insertion of unique models and their performance summary into the lstm_anomaly_models and lstm_anomaly_results table.
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-7-8-5-c450f35cf46e)


## Week 9&10


* Made certain changes in the configuration setup of the insight_worker.
* Wrote tests for InsightWorker class as weel as for the insight_model.
* Wrote documentation describing all the methods in the insight_model.
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-weekly-summary-week-9-10-6-622f17180d74)

## Week 11&12

* Comparative analysis of results of BiLSTM model and IsolationForest
* Wrote project summary
* [Notebook - BiLSTM_keras_model](https://github.com/pratikmishra356/GSOC-2020-Augur/blob/master/Notebooks/BiLSTM_keras_model.ipynb)
* [Summary](https://docs.google.com/document/d/1WBDsOHXtPJ9BlRSf7un9ennT6b5x4ngilU_smfpiorU/edit)
* [Blog](https://medium.com/@pratikmishra_60029/gsoc-2020-final-work-summary-7-4761f48ce6c8)
