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

