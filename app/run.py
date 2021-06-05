import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly import express as px
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # step 1 - Create a word cloud for the most common words in all the messages combined
    words = ""
    stopwords = set(STOPWORDS)

    for text in df.message:
        # remove punctuation and normalize text
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
        # apply tokenization
        tokens = word_tokenize(text)
    
        # lemmatize and remove stop words
        lemmatizer = WordNetLemmatizer() # define lemmatization algorithm
        tokens = [lemmatizer.lemmatize(word).strip() for word in tokens]
    
        words += " ".join(tokens) + " "
    
    # create word cloud
    word_cloud = WordCloud(width = 1600, height = 800, background_color ='white',
                           stopwords = stopwords, min_font_size = 10, colormap='inferno').generate(words)

    # step 2 - count the number of messages that fall into and don't fall into each of the
    # 36 message categories
    count_no = []
    count_yes = []

    # filter out columns for message categories that are required for step 2
    labels = df.columns.difference(['id', 'message', 'original', 'genre'])
    
    for col in df[labels].columns:
        # count the number of messages that fall into and don't fall into each category
        count_col = df[col].value_counts()
        
        # Store the number of messages that don't fall into the category
        count_no.append(count_col[0]) if 0 in count_col.index else count_no.append(0)
        # Store the number of messages that fall into the category
        count_yes.append(count_col[1]) if 1 in count_col.index else count_yes.append(0)
        
    # step 3 - Count the number of messages in each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals    
    wordcloud_fig = px.imshow(word_cloud)
    wordcloud_fig.update_layout(title=dict(text='Most Common Words in the Dataset', x=0.5),
                               xaxis={'showgrid':False, 'showticklabels':False, 'zeroline':False},
                               yaxis={'showgrid':False, 'showticklabels':False, 'zeroline':False},
                               hovermode=False)
    
    graphs = [
        {
            'data': [
                Bar(
                    name="Label 0",
                    x=labels,
                    y=count_no,
                   ),
                
                Bar(name="Label 1",
                    x=labels,
                    y=count_yes,
                   )
            ],
            
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                },
                'barmode': 'stack'
            }
        },

        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    graphs = [wordcloud_fig] + graphs
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()