from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import ImageTk, Image
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
import collections
from collections import Counter
import codecs
import sys
from enum import Enum

class Analyzer:
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words("english")
        self.x_graph = range(1, 26)

    def bargraph(self, namen, daten, titel):  # fuer die schönen Graphen
        fig = plt.figure()
        plt.bar(self.x_graph, daten, align="center")
        plt.xticks(self.x_graph, namen, rotation=45, ha='right')
        plt.title(titel)
        plt.tight_layout()
        plt.show()

    def read_csv(self, input_path):
        with codecs.open(input_path, "r", "utf-8") as csv_input:
            esa_reader = csv.reader(csv_input, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        return esa_reader

    def filter_words_keyword(self, data, keyword):
        result = []
        for column in data:
            for word in column[4].split():
                if keyword in word:
                    result.append(word)  # ranking aller begriffe mit keyword
        counted_results = Counter(result)
        counted_results_dict = dict(counted_results.most_common(25))
        results_keys = counted_results_dict.keys()
        results_values = counted_results_dict.values()
        return result, counted_results, results_keys, results_values

    def filter_words_stopwords(self, data):
        result = []
        for column in data:
            for word in column[4].split():
                if self.stopwords not in word:
                    result.append(word)  # ranking aller nicht stopworten
        counted_results = Counter(result)
        counted_results_dict = dict(counted_results.most_common(25))
        results_keys = counted_results_dict.keys()
        results_values = counted_results_dict.values()
        return result, counted_results, results_keys, results_values

    def filter_date(self, data):
        result = []
        for column in data:
            for datum in column[5].split():
                if Tokens.HYPHEN in datum:
                    result.append(datum)  # ranking aller begriffe mit keyword
        counted_results = Counter(result)
        counted_results_dict = dict(counted_results.most_common(25))
        results_keys = counted_results_dict.keys()
        results_values = counted_results_dict.values()
        return result, counted_results, results_keys, results_values

    def filter_tweets_date(self, data, search_date):
        result = []
        for column in data:
            for date in column[5]:
                if search_date in date:
                    result.append(date[4])  # ranking aller begriffe mit keyword
        return result

    def likelihood_keyword(self, data_one, data_two, keyword):
        result_words_one = []
        result_words_two = []
        result_same_words = []


        for column in data_one:
            for word in column[4].split():
                if keyword in word:
                    result_words_one.append(word)  # ranking aller nicht stopworten

        for column in data_two:
            for word in column[4].split():
                if keyword in word:
                    result_words_two.append(word)  # ranking aller nicht stopworten


        for word in result_words_one:
            if keyword in result_words_two:
                result_same_words.append(word)  # ranking aller nicht stopworten


        for word in result_words_two:
            if keyword in result_words_one:
                result_same_words.append(word)  # ranking aller nicht stopworten

        counted_results = Counter(result_same_words)
        counted_results_dict = dict(counted_results.most_common(25))
        results_keys = counted_results_dict.keys()
        results_values = counted_results_dict.values()

        return counted_results, results_keys, results_values



###############################################################################
#                                                                             #
#  List of some                                        #
#                                                                             #
###############################################################################
class Tokens(Enum):
    HASHTAG = "#"
    AT = '@'
    HYPHEN = '-'
