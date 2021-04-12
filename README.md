# Disaster Response Web Dashboard

## Table of Contents  
* [Project Overview](#overview)
* [Installation](#installation)  
* [File Descriptions](#file)
* [Results](#results)
* [Licensing, Authors, and Acknowledgements](#licensing)

<a id="overview"></a>
## Project Overview
The goal of this project is to create a web dashboard that disaster response organizations can use to help them in classifying a message into one or several categories using a trained machine learning (ML) model. There are 36 categories in total, and some examples of the categories are water, food and shelter. The web dashboard will also display visualizations of training data used to train the machine learning model.

To achieve the goal, a ETL pipeline was first created to prepare data for training machine learning models. Subsequently, a machine learning pipeline was defined to train and validate machine learning models with preprocessed training and validation data. Finally, a web dashboard with the aforementioned features was created.

<a id="installation"></a>
## Installation
Required libraries:

For ETL (process_data.py):
- sys
- pandas 
- sqlalchemy

For training machine learning model (train_classifier.py):
- sys
- pandas
- sqlalchemy
- re
- nltk
- sklearn
- joblib

For displaying web dashboard (run.py):
- json
- plotly
- plotly-express
- pandas
- nltk
- wordcloud
- re
- flask
- joblib
- sqlalchemy

After installing the libraries, the code should run with no issues using Python versions 3.*.

<a id="file"></a>
## File Descriptions


<a id="results"></a>
## Results
The main findings can be found at the Medium post available [here](https://data-science-novice.medium.com/what-influences-the-price-of-accommodations-on-airbnb-b7784b394330).
Your feedback is greatly appreciated.

<a id="licensing"></a>
## Licensing, Authors, and Acknowledgements
All credit to Airbnb for providing the data. The Licensing for the data and other descriptive information can be found at the 
Kaggle link available [here](https://www.kaggle.com/airbnb/seattle). Otherwise, feel free to use the code here as you would like!
