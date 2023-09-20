#!/usr/bin/env python
# coding: utf-8

# Train ML models on target(s) and save to disk
# * use feature extraction function from another locaton
# 
# * Model shows better results on test set when trained on newe data ~ 1998+ (better than 1980+, 1990+)
# * We could consider online training on added data if supported? skip for now
# 
# * Different function needed for loading this model and returning prediction
# 
# * I train 1 model for each different forecasting horizon, 1-7 Y, saved as f"{i+1}Y_model.cbm"


import pandas as pd
from catboost import CatBoostRegressor
pd.set_option('mode.use_inf_as_na', True)
from util_functions import *


def train_save_models(RAW_HISTORICAL_DATA_FILE = "full_training_data.csv",
    TARGET_COL = "norm_publications_count"):

    df = pd.read_csv(RAW_HISTORICAL_DATA_FILE)
    df = df.loc[df["Year"]>=1970]  # 1978
    df = df.sort_values(["Year","Term"],ascending=True).reset_index(drop=True)


    # In[4]:


    df = pipe.fit_transform(df)
    ## model works better when trained on later data
    df = df.loc[df["Year"]>=1992].reset_index(drop=True)


    # In[5]:


    for i in range(7):
        print(i+1)
        clf = CatBoostRegressor(verbose=False,has_time=True,ignored_features=["Term"],cat_features=["Term"]) # iterations=600,
        df["y"] = df.groupby("Term")[TARGET_COL].shift(-(i+1))
        
        X = df.dropna(subset="y",axis=0).drop(columns=['y'],errors="ignore").reset_index(drop=True).copy()
        y = df.dropna(subset="y",axis=0)["y"].reset_index(drop=True).copy()
        clf.fit(X,y)
        clf.save_model(f"{i+1}Y_model.cbm",format="cbm")


    # #### Below: Example for getting predictions
    # * Note: returns 2 types of predictions, with and without finetuning. (The latter may be leaky). Returns predictions for up to 6 years ahead by default

    # In[6]:


    # df_query = pd.read_csv(RAW_HISTORICAL_DATA_FILE)
    # df_query = df_query.loc[df_query["Year"]>=2011].sort_values(["Year","Term"],ascending=True).reset_index(drop=True)

    # my_res = get_forecast_predictions(df_query,max_forecast_range=6)
    # my_res


    # In[8]:


    # from create_tables_for_trend_prediction import create_term_table
    # my_res = get_forecast_predictions(create_term_table("neurotoxin"),max_forecast_range=4)


if __name__ == '__main__' :
    train_save_models()



