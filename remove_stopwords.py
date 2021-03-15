# Definig a function to remove the stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def remove_stopwords(text):

    words = [word for word in text if word not in stopwords.words('english')]
    return words
