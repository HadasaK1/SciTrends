from util_functions import get_forecast_predictions
from create_tables_for_trend_prediction import create_term_table
import os
import os.path
import datetime
#import sys



def save_data_with_prediction(term):
    print("save_data_with_prediction(term):")
    print(term)
    #get year
    today = datetime.date.today()
    year = today.year-1
    no_space_term=term.replace(" ","_")
    print("term.replace()")
    print(no_space_term)

    path="model_data/data_with_predictions_"+no_space_term+str(year)+".csv"
    print(path)
    if (os.path.isfile(path)):
        print("os.path.isfile(path)")
        return
    else:

        data = create_term_table(term)
        print("data")
        print(data)
        data_with_pred=data[['Term','Year', 'norm_publications_count']]
        data_with_pred.insert(3,"Data","real data")
        data_with_pred = data_with_pred[data_with_pred['Year'] <= year]
        print("data_with_pred")
        print(data_with_pred)


        predict = get_forecast_predictions(data)
        last_year_predict=predict.tail(1)
        print("last_year_predict")
        print(last_year_predict)

        p1 = [term,  year+1,  predict['pred_1Y'].iloc[-1],"prediction"]
        data_with_pred.loc[len(data_with_pred.index)]=p1
        p2 = [ term, year+2, predict['pred_2Y'].iloc[-1],"prediction"]
        data_with_pred.loc[len(data_with_pred.index)]=p2
        p3 = [ term,  year+3,  predict['pred_3Y'].iloc[-1],"prediction"]
        data_with_pred.loc[len(data_with_pred.index)]=p3
        p4 = [term, year+4,  predict['pred_4Y'].iloc[-1],"prediction"]
        data_with_pred.loc[len(data_with_pred.index)]=p4
        p5 = [ term, int(last_year_predict['Year'].iloc[0])+4, predict['pred_5Y'].iloc[-1],"prediction"]
        data_with_pred.loc[len(data_with_pred.index)]=p5
        p6 = [term, int(last_year_predict['Year'].iloc[0])+5,  predict['pred_6Y'].iloc[-1],"prediction"]
        data_with_pred.loc[len(data_with_pred.index)]=p6


        print("data_with_pred")
        print(data_with_pred)

        data_with_pred.to_csv(path)

#term="sanger sequencing"
#term=sys.argv[1]
#save_data_with_prediction(term)

