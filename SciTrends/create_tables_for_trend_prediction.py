# Import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import lxml
import lxml.etree
#import 'sys'



#####download the timeline table from Pubmed for a search string
def download_timeline_table(searchstr):
    # Create object page url
    URLbeg = "https://pubmed.ncbi.nlm.nih.gov/?term="
    url = URLbeg + searchstr

    # Create object page

    page = requests.get(url)

    # parser-lxml = Change html to Python friendly format
    # Obtain page's information
    # Obtain page's information
    soup = BeautifulSoup(page.text, 'lxml')
    print("dd")
    # Obtain information from tag <table>
    table1 = soup.find("table", id="timeline-table")
    print("ee")

    # Obtain every title of columns with tag <th>
    headers = []
    for i in table1.find_all("th"):
        title = i.text
        headers.append(title)

    # Create a dataframe
    term_data = pd.DataFrame(columns=headers)

    # Create a for loop to fill mydata
    for j in table1.find_all("tr")[1:]:
        row_data = j.find_all("td")
        row = [i.text for i in row_data]
        length = len(term_data)
        term_data.loc[length] = row
    term_data["Year"] = pd.to_numeric(term_data["Year"])
    term_data["Number of Results"] = pd.to_numeric(term_data["Number of Results"])
    return term_data


###create the table for prediction
def create_term_table(term):
    term_pub = download_timeline_table(term)
    print("term_pub")
    print(term_pub)
    #term_pub.to_csv("model_data/termmmmmmmmmmmmmmmmmm.csv")
    term_review_pub = download_timeline_table(
        term + " AND (review[Article Type] OR (systematic review)[Article Type])")
    print("term_review_pub")
    print(term_review_pub)
    #term_pub.to_csv("./model_data/termmmmmmmmmmmmmmmmmmreview.csv")
    # term_pub=download_timeline_table(term)
    term_df = pd.merge(term_pub, term_review_pub, on='Year', how='outer')
    term_df = term_df.replace(np.nan, 0)
    term_df.rename(columns={'Number of Results_x': 'publications_count',
                            'Number of Results_y': 'review_publications_count'}, inplace=True)
    term_df["Term"] = term

    # download general data
    general_pub = download_timeline_table("all[sb]")
    print("term_review_pub")
    # print(term_review_pub)
    #general_pub.to_csv("./model_data/generallllllllllllllllllll.csv")
    general_pub.rename(columns={'Number of Results': 'general_publication'}, inplace=True)

    # merge data
    term_general_df = pd.merge(term_df, general_pub, on='Year', how='left')

    # calculate normelaized publications count
    term_general_df["norm_publications_count"] = (term_general_df[
                                                      'publications_count'] /
                                                  term_general_df[
                                                      "general_publication"]) * 100000
    term_general_df["norm_review_publications_count"] = (term_general_df[
                                                             'review_publications_count'] /
                                                         term_general_df[
                                                             "general_publication"]) * 100000
    # remove the last roe of the last year (the last year may be biased)
    term_general_df = term_general_df[:-1]
    #term_general_df.to_csv("model_data/" + term + ".csv") #ORIG
    # #DAN: Uncommented + remove index:
    # term_general_df.to_csv("model_data/" + term + ".csv",index=False)
    return term_general_df

#data=create_term_table(term)
#data.to_csv("G:/My Drive/PhD/Trends/model_data/"+term+".csv")


def download_all_training_terms(terms_file= "training_terms_data.csv"):#"training_data_all.csv"):
    terms_list = pd.read_csv(terms_file)["Term"].unique() # [0:63]
    print(len(terms_list))
    all_terms_df = pd.DataFrame()
    for i,term in enumerate(terms_list):
        if i//5==0: print(i)
        res = create_term_table(term)
        # print(res)
        all_terms_df = pd.concat([all_terms_df,res])
    all_terms_df = all_terms_df.round(5).drop_duplicates().sort_values(by=["Year","Term"],ascending=True)
    all_terms_df = all_terms_df.loc[all_terms_df["Year"]>1920].reset_index(drop=True)
    print(all_terms_df.shape)
    print(all_terms_df.nunique())
    all_terms_df.to_csv("full_training_data.csv",index=False)

if __name__ =="__main__":
    download_all_training_terms()
