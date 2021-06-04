# Disaster Response Web Dashboard

## Table of Contents  
* [Project Overview](#overview)
* [Installation](#installation)  
* [File Descriptions](#file)
* [QuickStart](#quickstart)
* [Screenshots of Web Dashboard](#screenshots)
* [Licensing, Authors, and Acknowledgements](#licensing)


<a id="overview"></a>
## Project Overview
The goal of this project is to create a web dashboard that disaster response organizations can use to help them in classifying disaster messages into one or several categories using a machine learning (ML) model. There are 36 categories in total, and some examples of the categories are water, food, shelter, storm and aid_related. To classify a message, the user can simply enter the message in the provided text box. Subsequently, the dashboard will classify the message using the machine learning model and display the category(s). The web dashboard will also display visualizations of disaster data.

To achieve the goal, a ETL pipeline was first created to prepare data for training machine learning models. Subsequently, a machine learning pipeline was defined to train and validate machine learning models with preprocessed training and validation data. Finally, a web dashboard with the aforementioned features was created.

<a id="installation"></a>
## Installation
Required libraries:
* flask
* wordcloud
* plotly
* nltk
* pandas
* sqlalchemy
* sklearn
* joblib

After installing the libraries, the code should run with no issues using Python versions 3.*.

<a id="file"></a>
## File Descriptions
There are 3 main files for this project:
* run.py - to build a web dashboard, which can display visualizations of disaster data and classify disaster message, in a Flask web app
* process_data.py - to run an ETL pipeline that reads the dataset, clean the data, and then store the results in a SQLite database
* train_classifier.py - to run a ML pipeline that uses NLTK, scikit-learn's Pipeline and GridSearchCV to train and evaluate a machine learning model for classifying disaster messages
 
<a id="quickstart"></a>
## QuickStart
To run the web dashboard:
```
$ python app/run.py
```
and go to http://0.0.0.0:3001/

To run ETL pipeline
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

To run ML pipeline
```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

<a id="screenshots"></a>
## Screenshots of Web Dashboard


<a id="licensing"></a>
## Licensing, Authors, and Acknowledgements
All credit to Figure Eight for providing the data. Feel free to use the code here as you would like!
