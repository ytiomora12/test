import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()


slang_dict = pd.read_csv('new_kamusalay.csv', encoding = "ISO-8859-1")

def remove_symbols(text):
# Take a list of tokenized words as input
    text = [word.lower() for word in text if word.isalnum() and word.lower() != 'user' and word.lower() != 'rt']
    return text


def remove_stopwords(text):
    stopWords = set(stopwords.words('indonesian'))

    for w in text:
        if w in stopWords:
            text.remove(w)
    return text


slang_list = list(slang_dict['anakjakartaasikasik'])
def get_slang_index(slang):
    return slang_list.index(slang)

def standardize(text):
    for word in text:
        if word in slang_list:
            slang_index = get_slang_index(word)
            text[text.index(word)] = slang_dict['anak jakarta asyik asyik'][slang_index]
    return text

kata = 'USER ganteng ganteng lg gw sarap \xf0\x9f\x98\x82. Menyampaikan, mengatakan, disuruh'
tokens = word_tokenize(kata)
removed_symbol = remove_symbols(tokens)

standardized = standardize(removed_symbol)


def stemming(text):
    text_stemmed = []
    for word in text:
        word = stemmer.stem(word) # stemming word
        text_stemmed.append(word)
    return text_stemmed

stemming(standardized)


def clean_texts(text):
    
    tokenized = word_tokenize(text)
    removed_symbols = remove_symbols(tokenized)
    removed_stopwords = remove_stopwords(removed_symbols)
    standardized_words = standardize(removed_stopwords)
    stemmed_words = stemming(standardized_words)
    return stemmed_words
