# augur_ML_insights
https://github.com/chaoss/augur

![Screen Shot 2020-05-26 at 4 24 30 PM](https://user-images.githubusercontent.com/43684300/82900502-458e4680-9f7a-11ea-86fe-53532006ee8f.png)


This is not totally different from the current Insight_worker structure , I just added few more files to separate the preprocessing of the metrics and ML models from the worker.This will also be helpful for the new developers to integrate their ML models easily with the Insight_worker.

**preprocess_metrics.py** will select metric endpoints category wise and then it will create dataframe which will passed to the **ml_model.py** module where best fit ML_model will be applied and outliers will be inserted in the **insight_records_table**.Here we can also create top_insights metrics based on each model in the metric section of augur.**worker.py** will decide which outliers will be deleted or kept in insight_records_table based on the frequency , score , date etc.. and it will also send selected outliers to the augur's slack bot **"Auggie"** which will notify the user about the **most recent outliers**.








# Project Development 



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

