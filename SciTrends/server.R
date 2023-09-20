list.of.packages <- c("shiny", "DT","RColorBrewer","dplyr","shinydashboard","lubridate","magick","stringr","plotly","gapminder","reticulate")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# Import R packages needed for the app here:
library(shiny)
library(DT)
library(RColorBrewer)
library(dplyr)
library(shinydashboard)
library(lubridate)
library(magick)
library(stringr)
library(plotly)
library(gapminder)
library(reticulate)


# Define any Python packages needed for the app here:
PYTHON_DEPENDENCIES = c('catboost','pandas','beautifulsoup4','requests','numpy','lxml','datetime','scikit-learn','sentence_transformers')
# Begin app server
shinyServer(function(input, output) {
  
  # ------------------ App virtualenv setup (Do not edit) ------------------- #
  
  virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
  python_path = Sys.getenv('PYTHON_PATH')
  
  # Create virtual env and install dependencies
  reticulate::virtualenv_create(envname = virtualenv_dir, python = python_path)

  reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES, ignore_installed=TRUE)

  reticulate::use_virtualenv(virtualenv_dir, required = T)
  
  
  #____________________function for reading the data_________________________________#
  
  data_func <- function(term)
  { 
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    year<-year(Sys.Date())
    print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    reticulate::use_virtualenv(virtualenv_dir,required=TRUE)
    print("ccccccccccccccccccccccccccccccccc")
    reticulate::source_python("save_prediction.py")
    print("dddddddddddddddddddddddddddddddddd")
    tryCatch(
      {
        print("save_data_with_prediction_term_1")
        save_data_with_prediction(term)
        print("save_data_with_prediction_term_2")
        data<-read.csv(paste("./model_data/data_with_predictions_",str_replace_all(term, " ", "_"),year-1,".csv",sep=""))
        print(paste("./model_data/data_with_predictions_",str_replace_all(term, " ", "_"),year-1,".csv",sep=""))
        print(data)
        return(data)
      },
      #if an error occurs, tell me the error
      error=function(e)
      {
        message('An Error Occurred')
        print(e)
        return (data_frame(Year=c(),publications_count=c(),review_publications_count=c(),
                           Term=c(),general_publication=c(),norm_publications_count=c(),
                           norm_review_publications_count=c()))
      }
    )
  }
  
  
  # ------------------ App server logic  --------------- #
  
  data <- reactiveVal(
    data_frame(Year=c(),publications_count=c(),review_publications_count=c(),
               Term=c(),general_publication=c(),norm_publications_count=c(),
               norm_review_publications_count=c())
  )

  
  N<- reactiveVal(0)
  
  term<- reactiveVal("")
  
  
  
  observeEvent(input$go, {
    term(input$term)
    
    if (N()<10)
    {
      new_data<-data_func(input$term)
      print(nrow(new_data))
      print(ncol(new_data))
      
      if (file.exists(paste("model_data/data_with_predictions_",str_replace_all(input$term, " ", "_"),year(Sys.Date())-1,".csv",sep="")))
      {
        termm<-term()
        N(N()+1)
        #data(new_data)
        data(rbind(data(),new_data))
        data(data()[data()$Year>=1960,])
        #data(data()[data()$Year<=year(Sys.Date()),"Data"]<-"real data")
        
        output$error_text<-renderText({ 
          paste('\n\n',termm,"added to popularity graph",sep=" ")
        })
        
      }
      else 
        output$error_text<-renderText({ "\n\nNo data was found in PubMed for your search term"})
      
    }
    else
      output$error_text<-renderText({ 
        "\n\nYou can't add more then 10 search terms!
              You can remove one of the search terms and then add another term,
              or to open a the this website in a new tab."})
    
  })
  
  observeEvent(input$discard, {
    term(input$term)
    N(N()-1)
    if (nrow(data()[data()$Term==input$term,])!=0)
    {
      data(data()[data()$Term!=input$term,])
      output$error_text<-renderText({ 
        termm<-term()
        paste('\n\n',termm,"removed from popularity graph",sep=" ")
      })
    }
    else 
      output$error_text<-renderText({ 
        termm<-term()
        paste('\n\n',termm,"is not in the terms list",sep=" ")
      })
    
  })
  
  observeEvent(input$clear, {
    data(
      data_frame(Year=c(),publications_count=c(),review_publications_count=c(),
                 Term=c(),general_publication=c(),norm_publications_count=c(),
                 norm_review_publications_count=c()))
    
  })
  
  data_training <- reactiveVal(
    data_frame(Year=c(),publications_count=c(),review_publications_count=c(),
               Term=c(),general_publication=c(),norm_publications_count=c(),
               norm_review_publications_count=c())
  )
  
  
  
  training_term<- reactiveVal("")
  
  observeEvent(input$Training_term, {
    if (input$Training_term!="")
    {
    training_term(input$Training_term)
    
    new_training_data<-data_func(input$Training_term)
    print(nrow(new_training_data))
    print(ncol(new_training_data))
    print(paste("model_data/data_with_predictions_",str_replace_all(input$Training_term, " ", "_"),year(Sys.Date())-1,".csv",sep=""))
    if (file.exists(paste("model_data/data_with_predictions_",str_replace_all(input$Training_term, " ", "_"),year(Sys.Date())-1,".csv",sep="")))
    {
      data(rbind(data(),new_training_data))
      data(data()[data()$Year>=1960,])
      #data(data()[data()$Year<=year(Sys.Date()),"Data"]<-"real data")
      termm<-term()
      output$error_text<-renderText({ 
        paste('\n\n',termm,"added to popularity graph",sep=" ")})
        
    }
    else 
      output$error_text<-renderText({ "\n\nNo data was found in PubMed for your search term"})
  }})
  
  
  output$scatterplot <- renderPlotly({
    if(nrow(data())==0) return()
    
    
    req(data())
    
    vline <- function(x = 0, color = "green") {
      list(
        type = "line",
        y0 = 0,
        y1 = 1,
        yref = "paper",
        x0 = x,
        x1 = x,
        line = list(color = color, dash="dot")
      )
    }
    
    
    fig <- plot_ly(data(), x = ~Year,y = ~norm_publications_count, color=~Term, name=~paste(Term,Data,sep=" ")
                   ,symbol=~Data ,symbols = c('circle','o'),mode = 'markers',size=3) %>%
      layout(   yaxis = list(title = 'publications per 100,000'),shapes= list(vline(year(Sys.Date())-0.5)))
    
    
    
  })
  
  ###############################################Training data present###################
 
  
  

  
  

  
  
})