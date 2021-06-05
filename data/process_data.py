import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Load and merge two CSV files - one containing messages and the other containing categories
    
    Args:
         messages_filepath (str): Path to the CSV file containing messages
         categories_filepath (str): Path to the CSV file containing categories of each message

    Returns:
        df (DataFrame): A merged DataFrame containing messages and categories
    '''
    
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, on='id')
    
    return df
    
    
def clean_data(df):
    ''' 
    Clean the data for machine learning model. Cleaning processes include:
    1) Split 'categories' column in the dataframe into separate category columns.
    2) Convert category values to just numbers 0 or 1 by removing the texts.
    3) Replace 'categories' column in df with new category columns created in Step 1.
    4) Remove duplicates.
    5) Remove rows with 2 in 'related' category column.
    
    Args:
         df (DataFrame): A DataFrame

    Returns:
        df_clean (DataFrame): clean DataFrame
    '''
    
    # Make a copy of df
    df_clean = df.copy()
    
    # Create a dataframe of the 36 individual category columns
    categories = df_clean['categories'].str.strip().str.split(';', expand=True)
    # Select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # Use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[-1]
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # Drop the original categories column from 'df'
    df_clean = df_clean.drop(columns=['categories'])
    # Concatenate the original dataframe with the new 'categories' dataframe
    df_clean = pd.concat([df_clean, categories], axis=1)
    
    # Drop duplicates
    df_clean = df_clean.drop_duplicates()
    # Drop rows with 2 in 'related' column
    df_clean = df_clean[df_clean['related'] != 2].reset_index(drop=True)
    
    return df_clean
    
   
def save_data(df, database_filename):
    ''' Save clean dataset to a SQLite database
    
    Args:
        df (DataFrame): Clean dataframe
        database_filename (string): Path at which database will be stored
        
    Returns:
        None
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()