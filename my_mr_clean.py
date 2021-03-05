from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_content(article_name):
    source = requests.get('https://en.wikipedia.org/wiki/' + article_name).text
    soup = BeautifulSoup(source, 'lxml')
    article = soup.find('div', class_ = 'mw-parser-output')
    # print (article)
    return article

def merge_contents(data):
    data = data.find_all('p')
    text = ''
    for element in data:
        text += '\n' + ''.join(element.find_all(text = True))
    # print(text)
    return text

def tokenize(content):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    my_list = tokenizer.tokenize(content)
    # print(my_list)
    return my_list

def lower_collection(collection):
    for i in range(len(collection)):
        collection[i] = collection[i].lower()
    # print(collection)
    return collection

def count_frequency(collection):
    numpy_list = np.array(collection)
    (unique, counts) = np.unique(numpy_list, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    frequencies_table = pd.DataFrame(frequencies, columns=['word', 'frequency'])
    frequencies_table['frequency'] = frequencies_table['frequency'].apply(pd.to_numeric)
    frequencies_table = frequencies_table.sort_values('frequency', ascending = False, ignore_index = True)
    # print (frequencies_table)
    return frequencies_table

def print_most_frequent(frequencies, n):
    print (frequencies.head(n))
    
def vizualizing_all(df):
    df = df.sort_values('frequency').tail(20)
    df.plot.barh(x='word', y='frequency')
    plt.title('Most common Tokens in the Ozone layer article')
    plt.show()

stop_words = ["the", "of", "and", "in", "to", "is", "a", "by", "that", "are", "was", "from", "s", "as", "for", "about", "at", "it", "be", "on", "this", "these", "with", "which", "into", "have", "because", "5", "has", "all", "can", "most", "out", "other", "over", "9", "000", "b", "its", "an", "100", "were", "being", "been", "near", "very", "200", "c", "while", "than", "where"]

def remove_stop_words(words, stop_words):
    for i in range(len(words)):
        words[i] = words[i].lower()
    i = 0
    while i < len(words):
        for j in range(len(stop_words)):
            if (words[i] == stop_words[j]):
                words.pop(i)
                break
            elif (j == len(stop_words) - 1):
                i += 1
                break
    # print(words)
    # --- converting the list to dataframe for the conveniece of reviewer ---
    numpy_list = np.array(words)
    (unique, counts) = np.unique(numpy_list, return_counts=True)
    frequencies_without_stop_words = np.asarray((unique, counts)).T
    frequencies_table_without_stop_words = pd.DataFrame(frequencies_without_stop_words, columns=['word', 'frequency'])
    frequencies_table_without_stop_words['frequency'] = frequencies_table_without_stop_words['frequency'].apply(pd.to_numeric)
    frequencies_table_without_stop_words = frequencies_table_without_stop_words.sort_values('frequency', ascending = False, ignore_index = True)
    # print (frequencies_table_without_stop_words.head(20))
    return frequencies_table_without_stop_words

def vizualizing_without_stop_words(df):
    df = df.sort_values('frequency').tail(25)
    df.plot.barh(x='word', y='frequency')
    plt.title('Most common Tokens in the Ozone layer article (filtered)')
    plt.show()
    
data = get_content("Ozone_layer")
merge_content = merge_contents(data)
collection = tokenize(merge_content)
lower_collection = lower_collection(collection)
frequencies = count_frequency(lower_collection)
print_most_frequent(frequencies, 10)
vizualizing_all(frequencies)
filtered_collection = remove_stop_words(collection, stop_words)
vizualizing_without_stop_words(filtered_collection)
