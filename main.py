import gradio as gr
import numpy as np
import pandas as pd
from save_prediction import save_data_with_prediction
import datetime
import plotly.express as px



# global variables
data = pd.DataFrame()
last_term = ""
terms = []
shared_output = None

def create_plot(data):

    today = datetime.date.today()
    year = today.year - 1

    fig = px.scatter(data, x='Year', y='norm_publications_count', color='Term',
                     title='Scatter Plot with Color and Symbols')
    fig.update_layout(
        title="Outbreak in ",
        xaxis_title="year",
        yaxis_title="normalized publication count",
    )
    # Add a horizontal line at y=15
    fig.add_shape(type="line", x0=year + 0.5, x1=year + 0.5, y0=0, y1=data['norm_publications_count'].max(),
                  line=dict(color="blue", width=2, dash="dot"))

    return fig


def add_term_to_plot(custom_term,choice):

    global shared_output
    global last_term
    global terms
    global data  # Indicate that you want to work with the global 'data' variable
    term=""
    print("choice")
    print(choice)
    print("last_term")
    print(terms)

    if not custom_term or custom_term in terms:
        if not choice:
            #raise gr.Error("You didn't insert new term")
            return create_plot(data)
        if choice in terms:
            #raise gr.Error("Your choice is already in the graph")
            return create_plot(data)
        else:
            term = choice
            print ("term is choice "+term)

    elif not choice or choice in terms:
        if custom_term in terms:
            #raise gr.Error("The term you inserted is already in the graph")
            return create_plot(data)
        else:
            term = custom_term
            print ("term is custom_term "+term)


    #if both new
    if not term:
        #raise gr.Error("you inserted new terms in both options, the custom term is shown")
        term = custom_term
    print(term)

    if len(terms)>10:
        #raise gr.Error("The maximum terms number is 10")
        return create_plot(data)
    else:
        last_term = term
        terms = terms+[term]
        (print(terms))

        # get year
        today = datetime.date.today()
        year = today.year - 1
        no_space_term = term.replace(" ", "_")
        print(no_space_term)

        path = "model_data/data_with_predictions_" + no_space_term + str(year) + ".csv"
        try:
            save_data_with_prediction(no_space_term)
        except:
            raise gr.Error("There is no data about your term in Pubmed. "
                           "Please enter another term and the trend graph will be displayed")

        new_term_df = pd.read_csv(path)

        data = pd.concat([data, new_term_df], ignore_index=True)  # Concatenate DataFrames
        data = data[data['Year'] >= (year-40)]

        fig = create_plot(data)
        shared_output = fig
      #  update_choice2.update()
        return fig


def delete_term(term):

    global shared_output
    global terms
    global data

    if term in terms:
        terms.remove(term)
        data = data[data["Term"] != term]
        fig = create_plot(data)
        shared_output = fig
        return fig
    else:
        raise gr.Error(term + " is not exist in the graph!")

def clear_all():

    global shared_output
    global terms
    global data

    if len(terms) > 0:
        terms = []
        data = pd.DataFrame(columns=["Term","Year","norm_publications_count","Data"])
        fig = create_plot(data)
        shared_output = fig
        return fig
    else:
        raise gr.Error("The trends graph is already empty")
        return None



predefined_terms = [(term, term) for term in np.unique(pd.read_csv("training_data_all.csv")["Term"].to_numpy())]
description= "Welcome to the predictor of trends in science!\n\n\n" \
             "This tool predicts the popularity of fields in science for the next 6 years.\n" \
                 "Get predictions of scientific popularity for any term!\n"\
                 "Up to 10 terms can be inserted into the same graph.\n\n" \
                 "Popularity of a term is defined as the number of publications in PubMed per year for this term, \n" \
                 "normalized to 100, 000 publications..\n\n"\
                 "For details of the model and methodology see our paper. If you use us, please cite us!:\n"\
                "Ofer, D., & Linial, M. (2023). Whats next? Forecasting scientific research trends. ArXiv, abs/2305.04133.\n"\
                "Ofer et al., 2023. SciTrends [Internet].\n\n Available from:\n"\
                "https://393853b86b6381cc22.gradio.live\n\n"\
                "contact us at:\n"\
                "Hadasa.kaufman@mail.huji.ac.il\n\n\n"\
                "Developed by Hadasa Kaufman & Dan Ofer"

with gr.Blocks() as demo:
    # Create a Row component to divide the interface into two columns
    with gr.Row():
        # Create a Column for the left side (description)
        with gr.Column():
            gr.Image("logo_SciTrends.png")
            gr.Text(description,label="App Description")
        # Create a Column for the right side (input components and plot)
        with gr.Column():
            favicon="logo_SciTrends.png"  # Specify the path to your logo image
            txt1 = gr.components.Textbox(label="Insert a term")
            choice1 = gr.components.Dropdown(label="or choose an example term", choices=predefined_terms)
            btn = gr.Button(value="Submit")
            out1 = gr.Plot(label="plot")
            btn.click(add_term_to_plot, inputs=[txt1, choice1], outputs=out1 )
            remove_term = gr.components.Textbox(label="Insert a term to remove")
            btn = gr.Button(value="Remove term")
            btn.click(delete_term, inputs=remove_term, outputs=out1)
            btn = gr.Button(value="clear all")
            btn.click(clear_all, outputs=out1)





    live = True


if __name__ == "__main__":
    demo.launch(share=True)
