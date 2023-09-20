import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics

from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

#pd.set_option('mode.use_inf_as_na', True)


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('r2:', round(r2,3))
    print('MAE (Mean absolute error):', round(mean_absolute_error,3))
    print('Median absolute error:', round(median_absolute_error,3))
#     print('MSE: ', round(mse,3))
    print('RMSE:', round(np.sqrt(mse),3))
    print('explained_variance: ', round(explained_variance,3))    
    try:
        mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
        print('mape:', round(mape,3))
    except:()
    try: 
        mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
        print('mean_squared_log_error:', round(mean_squared_log_error,3))
    except:()
        

def run_eval(X_train,y_train,X_test,y_test):
    clf = CatBoostRegressor(verbose=0,iterations=600,has_time=True) # verbose=False,
    clf.fit(X_train,y_train)
#     clf.fit(X_train,y_train,eval_set=(X_test,y_test))#,plot=True)
    
    ### transforming the target with log - does better in earlier years, but orse in latter years, relative to not transforming
#     model_with_trans_target = TransformedTargetRegressor(
#         regressor=clf, func=np.log1p, inverse_func=np.expm1
#     ).fit(X_train, y_train)
#     print("\n log target")
#     regression_results(y_test, model_with_trans_target.predict(X_test))
    
    y_pred = clf.predict(X_test)

    regression_results(y_test, y_pred)


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, term_col, target_col=None, feature_cols=None):
        self.term_col = term_col
        self.target_col = target_col
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.sort_values(["Year"],ascending=True).reset_index(drop=True)# ensure ascending order
        X = X.copy()
        group = X.groupby(self.term_col)
        
        X["has_any_pubs_over2"] = (X[["publications_count","review_publications_count"]].max(axis=1)>2).astype(int)
        X["has_any_pubs_over2_window"] = group["has_any_pubs_over2"].shift(1).rolling(window=5, min_periods=1).sum()
        for col in self.feature_cols:
            # Calculate 1st and 2nd order difference features
            X[f'{col}_diff1'] = group[col].transform(lambda x: x.diff())
            X[f'{col}_diff2'] = group[f'{col}_diff1'].transform(lambda x: x.diff())

            # Calculate percent change features
            X[f'{col}_pct_change'] = group[col].transform(lambda x: x.pct_change())
            X[f'{col}_pct_change_diff1'] = group[f'{col}_pct_change'].transform(lambda x: x.diff())
            
            X[f'{col}_expanding_max'] = group[col].transform(lambda x: x.shift().expanding().max())
            X[f'{col}_rolling_max_5'] = group[col].transform(lambda x: x.shift().rolling(window=5, min_periods=1).max())
            X[f'{col}_rolling_min_5'] = group[col].transform(lambda x: x.shift().rolling(window=5, min_periods=1).min())
            X[f'{col}_rolling_avg_5'] = group[col].transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
            X[f'{col}_rolling_sum_5'] = group[col].transform(lambda x: x.shift().rolling(window=5, min_periods=1).sum())
            X[f'{col}_rolling_median_12'] = group[col].transform(lambda x: x.shift().rolling(window=12, min_periods=1).median())
            X[f'{col}_rolling_max_15'] = group[col].transform(lambda x: x.shift().rolling(window=15, min_periods=1).max())
            X[f'{col}_rolling_avg_10'] = group[col].transform(lambda x: x.shift().rolling(window=10, min_periods=1).mean())
            X[f'{col}_rolling_ema_15'] = group[col].transform(lambda x: x.shift().ewm(span=15, adjust=False).mean())

            
            X[f'{col}_lag1'] = group[col].shift(1)
            X[f'{col}_lag2'] = group[col].shift(2)

            X[f'{col}_pct_change_lag1'] = group[f'{col}_pct_change'].shift(1)
            X[f'{col}_pct_change_lag2'] = group[f'{col}_pct_change'].shift(2)
            X[f'{col}_diff1_lag1'] = group[f'{col}_diff1'].shift(1)
            X[f'{col}_diff1_lag2'] = group[f'{col}_diff1'].shift(2)


            X[f'{col}_diff2_lag1'] = group[f'{col}_diff2'].shift(1)
            X[f'{col}_diff2_lag2'] = group[f'{col}_diff2'].shift(2)
            X[f'{col}_pct_change_diff1_lag1'] = group[f'{col}_pct_change_diff1'].shift(1)
            X[f'{col}_pct_change_diff1_lag2'] = group[f'{col}_pct_change_diff1'].shift(2)
            X[f'{col}_pct_change_diff1_lag3'] = group[f'{col}_pct_change_diff1'].shift(3)
            ### extract deeper features for the target column
            if col in self.target_col:
                X[f'{col}_lag3'] = group[col].shift(3)
                X[f'{col}_lag5'] = group[col].shift(5)
                X[f'{col}_lag8'] = group[col].shift(8)
                X[f'{col}_lag11'] = group[col].shift(11)


                X[f'{col}_pct_change_lag3'] = group[f'{col}_pct_change'].shift(3)
                X[f'{col}_pct_change_lag5'] = group[f'{col}_pct_change'].shift(5)
                X[f'{col}_pct_change_lag7'] = group[f'{col}_pct_change'].shift(7)
                X[f'{col}_pct_change_lag11'] = group[f'{col}_pct_change'].shift(11)

                X[f'{col}_diff1_lag5'] = group[f'{col}_diff1'].shift(5)

                X[f'{col}_pct_change_diff1_lag5'] = group[f'{col}_pct_change_diff1'].shift(5)

                X[f'{col}_diff2_lag3'] = group[f'{col}_diff2'].shift(3)
                X[f'{col}_diff2_lag4'] = group[f'{col}_diff2'].shift(4)

                # Calculate max, min, average, EMA over the past 5 years
                X[f'{col}_rolling_ema_5'] = group[col].transform(lambda x: x.shift().ewm(span=5, adjust=False).mean())

                # Calculate max, min, average, EMA over all past history

                X[f'{col}_expanding_min'] = group[col].transform(lambda x: x.shift().expanding().min())
                X[f'{col}_expanding_avg'] = group[col].transform(lambda x: x.shift().expanding().mean())
                X[f'{col}_expanding_ema'] = group[col].transform(lambda x: x.shift().ewm(com=0.5, adjust=False).mean())

                ### history for derived cols

                X[f'{col}_diff1_rolling_max_8'] = group[f'{col}_diff1'].transform(lambda x: x.shift().rolling(window=8, min_periods=1).max())
                X[f'{col}_diff1_rolling_min_8'] = group[f'{col}_diff1'].transform(lambda x: x.shift().rolling(window=8, min_periods=1).min())
                X[f'{col}_diff1_rolling_avg_8'] = group[f'{col}_diff1'].transform(lambda x: x.shift().rolling(window=8, min_periods=1).mean())
                X[f'{col}_diff1_rolling_ema_12'] = group[f'{col}_diff1'].transform(lambda x: x.shift().ewm(span=12, adjust=False).mean())

                X[f'{col}_diff2_rolling_max_10'] = group[f'{col}_diff2'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).max())
                X[f'{col}_diff2_rolling_min_10'] = group[f'{col}_diff2'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).min())
                X[f'{col}_diff2_rolling_avg_10'] = group[f'{col}_diff2'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).mean())

                X[f'{col}_pct_change_rolling_max_10'] = group[f'{col}_pct_change'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).max())
                X[f'{col}_pct_change_rolling_min_10'] = group[f'{col}_pct_change'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).min())
                X[f'{col}_pct_change_rolling_avg_10'] = group[f'{col}_pct_change'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).mean())

                X[f'{col}_pct_change_positives_6'] = group[f'{col}_pct_change'].transform(lambda x: x.shift().rolling(window=6, min_periods=1).agg(lambda x: x[x>0.03].sum()))

                X[f'{col}_pct_change_diff1_rolling_max_7'] = group[f'{col}_pct_change_diff1'].transform(lambda x: x.shift().rolling(window=7, min_periods=1).max())
                X[f'{col}_pct_change_diff1_rolling_min_7'] = group[f'{col}_pct_change_diff1'].transform(lambda x: x.shift().rolling(window=7, min_periods=1).min())
                X[f'{col}_pct_change_diff1_rolling_avg_7'] = group[f'{col}_pct_change_diff1'].transform(lambda x: x.shift().rolling(window=7, min_periods=1).mean())
                X[f'{col}_pct_change_diff1_rolling_median_12'] = group[f'{col}_pct_change_diff1'].transform(lambda x: x.shift().rolling(window=12, min_periods=1).median())


# #         # y - The target variables (3 years and 5 years into the future)
#         if self.target_col:
#             X['target_3_years'] = group[self.target_col].shift(-3)

              
        #### interaction cols - reviews to research ratio/diff
        X["review_research_ratio"] = X["review_publications_count"].div(X["publications_count"]).fillna(0)
        X["research_review_norm_diff"] = X["norm_publications_count"].fillna(0).sub(X["norm_review_publications_count"].fillna(0))
        
        X["review_research_ratio_mean4"] = group["review_research_ratio"].transform(lambda x: x.shift().rolling(window=4, min_periods=1).mean())
        X["review_research_ratio_max6"] = group["review_research_ratio"].transform(lambda x: x.shift().rolling(window=6, min_periods=1).max())
        X["research_review_norm_diff_mean5"] = group["research_review_norm_diff"].transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
        X["research_review_norm_diff_max8"] = group["research_review_norm_diff"].transform(lambda x: x.shift().rolling(window=8, min_periods=1).max())
        X["research_review_norm_diff_min6"] = group["research_review_norm_diff"].transform(lambda x: x.shift().rolling(window=6, min_periods=1).min())
        ## year based features
        
        X["time_since_first_occ_year"] = X["Year"].sub(X.loc[X["publications_count"]>2].groupby("Term")["Year"].transform("first"))
        X["time_since_first_rev_occ_year"] = X["Year"].sub(X.loc[X["review_publications_count"]>2].groupby("Term")["Year"].transform("first") )
        
        
        #### Add DL embedding features for term. Can drop Term after this
        model = SentenceTransformer('all-MiniLM-L12-v2')#,device="cpu") #
        # Terms/Topics/sentences we want to encode
        sentences = list(X["Term"].unique())

        #Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)

        df_embed_feats = pd.DataFrame(index=sentences, data=embeddings)
        df_embed_feats.columns = ["embed_"+str(i) for i in df_embed_feats.columns]
        X = X.join(df_embed_feats,on="Term",how="left")
        
        # Dropping the first and last few rows for each group as these rows will have NaN values
#         X = X.dropna() ## orig
        ## new dropna - target subset?  # change for target!!
#         X = X.dropna(subset=['target_3_years'])
        X = X.dropna(subset=[self.target_col])

        return X.sort_values(["Year"]).reset_index(drop=True)

    
# Apply the transformations to the DataFrame
pipe = Pipeline([
    ('ts_features', TimeSeriesFeatureExtractor(
        term_col='Term', 
        target_col='norm_publications_count', 
        feature_cols=[#'publications_count', 'review_publications_count',
                      'norm_publications_count',
#                       'norm_review_publications_count'
                     ])),
])


def get_forecast_predictions(df_query:pd.DataFrame,max_forecast_range=6,TARGET_COL='norm_publications_count'):
    df_query = df_query.loc[df_query["Year"]>=1980].sort_values(["Year","Term"],ascending=True).reset_index(drop=True)
    df_query = pipe.transform(df_query)

    ## model works better when trained on later data
    df_query = df_query.loc[df_query["Year"]>=2001].reset_index(drop=True)
    
    df_res = df_query[["Term","Year",TARGET_COL]].copy()
    for i in range(max_forecast_range):

        model = CatBoostRegressor()
        model.load_model(f"{i+1}Y_model.cbm")
        
        y_pred = model.predict(df_query)
        
        ### finetuned model preds - optional
        y_test = df_query.groupby("Term")[TARGET_COL].shift(-(i+1))
        X_test = df_query.copy()

        mask = y_test.notna()
        y_test = y_test[mask]
        X_test = X_test[mask]

#         print("extra train?, on query data")
        model2 = CatBoostRegressor(iterations=10,learning_rate=0.1,verbose=False,has_time=True)
        model2.fit(X_test,y_test,init_model=model,cat_features=["Term"])
        y_pred_ft = model2.predict(df_query)
        
        df_res[f"pred_{i+1}Y"] = y_pred
        df_res[f"pred_ft_{i+1}Y"] = y_pred_ft
    return df_res
        
