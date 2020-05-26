# augur_ML_insights
https://github.com/chaoss/augur

![Screen Shot 2020-05-26 at 4 24 30 PM](https://user-images.githubusercontent.com/43684300/82900502-458e4680-9f7a-11ea-86fe-53532006ee8f.png)


This is not totally different from the current Insight_worker structure , I just added few more files to separate the preprocessing of the metrics and ML models from the worker.This will also be helpful for the new developers to integrate their ML models easily with the Insight_worker.

**preprocess_metrics.py** will select metric endpoints category wise and then it will create dataframe which will passed to the **ml_model.py** module where best fit ML_model will be applied and outliers will be inserted in the **insight_records_table**.Here we can also create top_insights metrics based on each model in the metric section of augur.**worker.py** will decide which outliers will be deleted or kept in insight_records_table based on the frequency , score , date etc.. and it will also send selected outliers to the augur's slack bot **"Auggie"** which will notify the user about the **most recent outliers**.
