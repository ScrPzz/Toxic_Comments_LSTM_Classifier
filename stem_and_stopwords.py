
import string  # for string operations

from nltk.corpus import stopwords  # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # module for stemming

from nltk.tokenize import TreebankWordTokenizer


def stem_and_stopwords(comment):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tokenizer = TreebankWordTokenizer()
    comment_tokens = tokenizer.tokenize(comment)
    # print(comment_tokens)
    comments_clean = []
    for word in comment_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            comments_clean.append(stem_word)
    return comments_clean
