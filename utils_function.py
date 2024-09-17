import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re  # Import regular expressions library
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def plot_categorical_proportions(df):
    """
    Plots bar charts for each categorical variable in a DataFrame, showing the proportion of each category,
    ordered by proportion in descending order. Each bar is labeled with its percentage value.

    Inputs:
    df (pd.DataFrame): The DataFrame to analyze.

    Outputs:
    None

    Description:
    This function identifies categorical variables, calculates the proportion of each category, sorts them,
    and plots a bar chart for each categorical variable. Labels on the bars display the percentage proportion of each category,
    excluding the 'tweet_text' column.
    """
    # Excluding 'tweet_text' column
    df = df.drop(columns=['tweet_text'])
    
    for col in df.columns:
        # Calculating proportions
        value_counts = df[col].value_counts(normalize=True).sort_values(ascending=False)
        percentages = value_counts * 100  # Convert proportions to percentages
        
        # Plotting
        plt.figure(figsize=(10, 6))
        ax = percentages.plot(kind='bar')
        ax.set_title(f'Proportion of Categories in {col}')
        ax.set_ylabel('Percentage')
        
        # Adding percentage labels on the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        plt.show()
        
        
def plot_grouped_charts(df):
    """
    Creates combined plots for each column in the DataFrame based on their data type, grouped by the 'is_there_an_emotion_directed_at_a_brand_or_product' column.
    For numeric columns, histograms for all statuses are combined in one plot, and boxplots for all statuses are combined in another.
    For categorical columns, grouped bar charts are created.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    """
    status_col = 'is_there_an_emotion_directed_at_a_brand_or_product'
    unique_statuses = df[status_col].unique()
    colors = plt.get_cmap('tab10')  # Fetches a colormap with distinct colors

    for col in df.columns:
        if col not in [status_col, 'tweet_text']:
            if df[col].dtype in ['int64', 'float64']:  # Numeric Columns
                plt.figure(figsize=(12, 6))
                
                # Histogram for all statuses
                for i, status in enumerate(unique_statuses):
                    sns.histplot(df[df[status_col] == status][col], kde=True, element='step', 
                                 stat='density', label=str(status), color=colors(i))
                
                plt.title(f'Combined Histogram of {col} by {status_col}')
                plt.legend(title=status_col)
                plt.show()
                
                # Boxplot for all statuses
                plt.figure(figsize=(12, 6))
                sns.boxplot(x=status_col, y=col, data=df, palette='tab10')
                plt.title(f'Combined Boxplot of {col} by {status_col}')
                plt.show()
            
            elif df[col].dtype == 'object':  # Categorical Columns
                plt.figure(figsize=(10, 6))
                sns.countplot(data=df, x=status_col, hue=col)
                plt.title(f'Grouped Bar Chart of {status_col} by {col}')
                plt.ylabel('Count')
                plt.xlabel(status_col)
                plt.legend(title=col, loc='upper right')
                plt.xticks(rotation=45)
                plt.show()

          
def txt_clean(txt):
    """
    Clean and preprocess text data for further analysis.

    Parameters:

        txt (str): The text string that needs to be cleaned and tokenized.

    Returns:

        list: A list of cleaned and tokenized words, where punctuation and special characters are replaced by spaces, text 
        is converted to lowercase, stopwords and Twitter mentions are removed, words with accents are excluded, and empty 
        strings are filtered out.
    
    """
    
    # List of additional strange characters to remove
    strange_chars = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~“!#Ûª'
    
    sw = stopwords.words('english')
    sw.extend(['link', 'rt', 'get'])
    no_accents_re = re.compile('^[a-z]+$')
    twitter_re = re.compile('[@][a-zA-Z]*')

    # Replace punctuation and strange characters with spaces
    txt = txt.translate(str.maketrans(strange_chars, ' ' * len(strange_chars)))

    # Tokenize the text
    tokens = word_tokenize(txt)

    # Convert to lowercase
    tokens = [w.lower() for w in tokens]
    # Remove @ mentions
    tokens = [w for w in tokens if not twitter_re.match(w)]
    # Remove words with accents
    tokens = [w for w in tokens if no_accents_re.match(w)]
    # Remove stopwords
    tokens = [w for w in tokens if w not in sw]
    # Remove empty strings
    tokens = [w for w in tokens if w]

    return tokens


# Define the function to identify if the tweet is about a Google or Apple product
def identify_product(tweet_text):
    """
    Identify if the tweet is about a Google or Apple product.
    
    Parameters:
    tweet_text (str): The text of the tweet.
    
    Returns:
    str: 'Google' if the tweet mentions a Google product, 'Apple' if the tweet mentions an Apple product,
         'Both' if the tweet mentions both, 'Unknown' if it mentions neither.
    """
    google_keywords = ['google', 'pixel', 'pixels', 'nexus', 'nexuses', 'android', 'androids', 
                       'chromebook', 'chromebooks', 'nest', 'nests', 'stadia', 'stadias']
    apple_keywords = ['apple', 'apples', 'iphone', 'iphones', 'ipad', 'ipads', 'macbook', 
                      'macbooks', 'imac', 'imacs', 'watch', 'watches', 'airpods', 
                      'appstore', 'ios', 'itunes']
    
    # Ensure tweet_text is a string
    if not isinstance(tweet_text, str):
        return 'Unknown'
    
    # Replace "app store" with "appstore" before tokenization
    tweet_text = tweet_text.replace("app store", "appstore")
    
    # Remove all numbers from the tweet text
    tweet_text = re.sub(r'\d+', '', tweet_text)
    
    # Clean the text and obtain tokens
    tokens = txt_clean(tweet_text)
    
    # Check if any keyword exists as a substring within the tokens
    google_mentioned = any(any(keyword in token for keyword in google_keywords) for token in tokens)
    apple_mentioned = any(any(keyword in token for keyword in apple_keywords) for token in tokens)
    
    if google_mentioned and apple_mentioned:
        return 'Both'
    elif google_mentioned:
        return 'Google'
    elif apple_mentioned:
        return 'Apple'
    else:
        return 'Unknown'
    

def plot_emotion_distribution(df, product_column='product_mention', emotion_column='emotion_type'):
    """
    Plots a bar chart showing the distribution of emotion types by product mention,
    with annotations displaying the counts and percentages.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    product_column (str): The name of the column that contains product mentions. Default is 'product_mention'.
    emotion_column (str): The name of the column that contains emotion types. Default is 'emotion_type'.
    """
    # Calculate the counts and normalize to get percentages
    counts = df.groupby([product_column, emotion_column]).size().reset_index(name='counts')
    total_counts = df[product_column].value_counts().reset_index()
    total_counts.columns = [product_column, 'total']

    # Merge counts with totals to calculate percentages
    counts = counts.merge(total_counts, on=product_column)
    counts['percentage'] = (counts['counts'] / counts['total']) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=counts, x=product_column, y='percentage', hue=emotion_column)

    # Annotate each bar with the corresponding count and percentage
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)} ({height:.1f}%)', 
                    xy=(p.get_x() + p.get_width() / 2, height), 
                    xytext=(0, 5),  # Offset label position above the bar
                    textcoords='offset points',
                    ha='center', va='center')
    
    plt.title('Distribution of Emotion Types by Product Mention')
    plt.ylabel('Percentage')
    plt.xlabel('Product Mention')
    plt.xticks(rotation=45)
    plt.legend(title=emotion_column)
    plt.show()
    

def get_contingency_table_with_percentage_sign(df, product_column='product_mention', emotion_column='emotion_type'):
    """
    Generates a contingency table for the given DataFrame based on product mentions and emotion types,
    showing the proportions as percentages with a percentage sign, rounded to 2 decimal places.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    product_column (str): The name of the column that contains product mentions. Default is 'product_mention'.
    emotion_column (str): The name of the column that contains emotion types. Default is 'emotion_type'.

    Returns:
    pd.DataFrame: A contingency table with proportions (as percentages) of emotion types for each product mention.
    """
    # Create a contingency table with counts
    contingency_table = pd.crosstab(df[product_column], df[emotion_column])

    # Convert counts to proportions by dividing each cell by the row sum and multiply by 100 to get percentages
    contingency_table_percentage = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

    # Round the percentages to 2 decimal places and add the percentage sign
    contingency_table_percentage = contingency_table_percentage.round(2).astype(str) + '%'

    return contingency_table_percentage



def tweet_text_treatment(df, text_column='tweet_text'):
    """
    This function processes the text in a specified column of a dataframe by applying several steps:
    1. Lowercasing all text.
    2. Replacing product names, user tags, hashtags, and URLs with general terms.
    3. Removing stopwords.
    4. Removing strange characters and punctuation.
    5. Removing numbers.
    6. Eliminating single-letter words.
    7. Lemmatizing words.
    8. Tokenizing the text.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text data to process.
    text_column (str): The name of the column in the dataframe that contains the text to be processed.

    Returns:
    pd.DataFrame: DataFrame with processed text and a new column for tokenized text.
    """

    # Step 1: Convert all text to lowercase
    df[text_column] = df[text_column].str.lower()

    # Step 2: Replace product names, user tags, hashtags, and URLs with general terms
    df[text_column] = df[text_column].map(identify_product)

    # Step 3: Remove stopwords from the text
    stopwords_to_remove = stopwords.words('english')
    df[text_column] = df[text_column].map(lambda x: ' '.join([word for word in x.split() if word not in stopwords_to_remove]))

    # Step 4: Remove strange characters and punctuation
    strange_chars = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~“!#Ûªåçûïòóêâîô¼¾½±°¤¦¬áàâäãåéèêëíìîïóòôöõúùûüýÿđøñìùđūöý'
    df[text_column] = df[text_column].map(lambda x: x.translate(str.maketrans(strange_chars, ' ' * len(strange_chars))))

    # Step 5: Remove numbers from the text
    df[text_column] = df[text_column].map(lambda x: re.sub(r'\d+', '', x))

    # Step 6: Eliminate single-letter words from each tweet
    df[text_column] = df[text_column].map(lambda x: ' '.join([word for word in x.split() if len(word) > 1]))

    # Step 7: Initialize the WordNet lemmatizer and lemmatize each word in the text
    lemmatizer = WordNetLemmatizer()
    df[text_column] = df[text_column].map(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    # Step 8: Tokenize the processed text
    df[f'{text_column}_tokenized'] = df[text_column].map(lambda x: word_tokenize(x))

    return df


