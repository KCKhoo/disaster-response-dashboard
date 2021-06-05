import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'stopwords', 'wordnet'])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib


def load_data(database_filepath):
    ''' Load dataset from SQLite database and split it into X (messages) and Y (category labels)
    
    Args:
         database_filepath (str): Path to the SQLite database

    Returns:
        X (ndarray): An array of text messages
        Y (ndarray): A two-dimensional array of category labels
        category_names: A list of category names
    '''
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterMessages', engine)

    # Extract text messages
    X = df['message'].values

    # Extract category labels and category names
    category_names = list(df.columns.difference(['id', 'message', 'original', 'genre']))
    Y = df[category_names].values
    
    return  X, Y, category_names


def tokenize(text):
    '''
    Return a list of tokens after removing punctuation from text, followed by 
    applying normalization, tokenization and lemmatization, and removing stopwords

    Args:
        text (str): A string literal 

    Returns:
        tokens (list): A list of tokens
    '''
    
    # Remove punctuation and normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Apply tokenization
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer() # define lemmatization algorithm
    stop_words = stopwords.words("english") # define common English stop words
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    '''
    Return a machine learning pipeline that contains all the preprocessing steps and a
    multi-output classifier for training and testing. For training, hyperparameters
    tuning is also performed using grid search with cross validation
   
    Args:
        None

    Returns:
       cv: A machine learning pipeline for training and testing
    '''
    
    # Build machine learning pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier())),
                        ])
    
    # Define values of hyperparameters to be searched through
    parameters = {
                  'clf__estimator__n_estimators': [50, 100, 150],
                  'clf__estimator__min_samples_split': [3, 4],
                 }
    
    # Grid search with cross validation
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluate model with F1 score, precision and recall
    
    Args:
        model: Trained machine learning model
        X_test (ndarray): An array of text messages
        Y_test (ndarray): A two-dimensional array of category labels
        category_names: A list of category names

    Returns:
       None
    '''
    
    # Predict outputs of test sets with the trained model
    Y_pred = model.predict(X_test)

    # Evaluate performance of trained model with F1 score, precision and recall for each output category
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    ''' Save final model as a pickle file
    
    Args:
        model: Final machine learning model with the best performance
        model_filepath (string) : Path at which the final model will be stored 

    Returns:
       None
    '''
    
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()