# Import R packages needed for the UI
library(shiny)
library(shinycssloaders)
library(DT)
library(dplyr)
library (rio)
library(shinydashboard)
library(lubridate)
library(magick)
library(stringr)
library(plotly)
library(gapminder)
library(reticulate)
library(shinyWidgets)


# Begin UI for the R + reticulate example app
ui <- fluidPage(
  

  sidebarLayout(
    
    # ---------------- Sidebar panel with changeable inputs ----------------- #
    titlePanel( img(src = "logo_SciTrends.png",height=150,width=550)),
    
    sidebarLayout(
      
      sidebarPanel(
        
        h3("Welcome to the predictor of trends in science!"),
        
        
        pickerInput(
          inputId = "Training_term",
          label = "Example Terms",
          selected="",
          choices = sorted_vector <- c("",sort(unique(read.csv("training_data_all.csv")$Term)))
          ,
          options = list(
            "actions-box" = TRUE,  # Show checkboxes for multiple selections
            "live-search" = TRUE  # Enable live search
          ),
          multiple = FALSE
        ),
        textInput(inputId = "term",
                  label = "Enter term",
                  value = ""),
        
        
        actionButton("go", "  Show prediction ", icon("rotate-right"), 
                     style="color: #fff; background-color: #337ab7; border-color: #2e6da4"),
        p(),
        
        actionButton("discard", "Remove term",icon("trash"), style='padding:4px; font-size:80%'),
        p(),
        actionButton("clear", "Clear all",icon("trash"), style='padding:4px; font-size:80%'),
        p(),
        
        
        p("This tool predicts the popularity of fields in science for the next 6 years."),
        p("Get predictions of scientific popularity for any term!"),
        p('Up to 10 terms can be inserted into the same graph. Terms can be deleted from the graph using the "Remove prediction" button.'),
        p(),
        p(),
        p(),
        fluidRow(
          textOutput(outputId= "error_text"),
          tags$head(tags$style("#error_text{color: blue;
                                 font-size: 20px;
                                 font-style: italic;
                                 }"))
        ),
        
        
      
p("*Popularity of a term is defined as the number of publications in PubMed per year for this term, normalized to 100,000 publications."),
        
        
        p("For details of the model and methodology see our paper. If you use us, please cite us!:"),
        p("Ofer, D., & Linial, M. (2023). Whats next? Forecasting scientific research trends. ArXiv, abs/2305.04133."),
        p("Offer et al., 2023. SciTrends [Internet].\n Available from:"),
        a("https://hadasakaufman.shinyapps.io/SciTrend/"),
        
        p("contact us at:"),
        a("Hadasa.kaufman@mail.huji.ac.il"),
        
        p("Developed by Hadasa Kaufman & Dan Ofer")
        
        
      ),
      
      

      
      
    
    # ---------------- Sidebar panel with changeable inputs ----------------- #
    
    mainPanel(
      titlePanel( img(src = "under_construction_messege.png",height=200,width=700)),
      
      h3("Trends Over Time Results"),
      shinycssloaders::withSpinner(
        plotlyOutput(outputId = "scatterplot")),
      
    

      
    )
    )
  )
)