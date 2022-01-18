import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
import string
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

def preprocess(listToProcess):
    '''removes stopwords, punctuation, numbers and single letters. Lowercases and counts every word.
    Code found from https://songxia-sophia.medium.com/word-embedding-of-brown-corpus-using-python-ec09ff4cbf4f'''
    
    lower_words = [x.lower() for x in listToProcess]
    pun_stop = list(string.punctuation) + stopwords.words('english')
    filter_words1 = [x for x in lower_words if x not in pun_stop]
    filter_words = list(filter(lambda x: x.isalpha() and len(x)>1, filter_words1))
    
    return filter_words
    
def lemmatize(listToProcess):
    lemmatized_list = []
    wnl = WordNetLemmatizer()
    
    for word in listToProcess:
        lemmatized_list.append(wnl.lemmatize(word))
    
    return lemmatized_list
    
    