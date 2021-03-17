from typing import List, NamedTuple
from tokenizer import dumb_tokenize
from collections import defaultdict
import numpy as np

# https://stats.stackexchange.com/questions/142505/how-to-use-naive-bayes-for-multi-class-problems
""" code adapted from 
    Data Science from Scratch, Second Edition, by Joel Grus (O’Reilly). 
    Copyright 2019 Joel Grus, 978-1-492-04113-9.
"""
class NaiveBayesClassifier:
    def __init__(self, categories, smoothing_factor):
        self.probs = {}
        self.categories = categories
        self.category_count = {category:0 for category in self.categories}
        self.datapoints_count = 0
        self.pos = 'pos'
        self.neg = 'neg'
        self._initialize_counters()
        self.smoothing_factor = smoothing_factor
        pass
    

    def _initialize_counters(self):
        for category in self.categories:
            self.probs[category] = {}
            self.probs[category][self.pos] = defaultdict(int)
            self.probs[category][self.neg] = defaultdict(int)


    def train(self, dataset:List[NamedTuple]) -> None:
        self.datapoints_count = len(dataset)
        for datapoint in dataset:
            self.category_count[datapoint.category] += 1
            for token in dumb_tokenize(datapoint.text):
                self.probs[datapoint.category][self.pos][token] += 1
                for category in self.categories:
                    if category != datapoint.category:
                        self.probs[category][self.neg][token] += 1
        pass

    def _calculate_probabilities(self, token:str, category:str) -> float:
        prob_if_token_given_category = (self.smoothing_factor + self.probs[category][self.pos][token])/(self.category_count[category] + 2 * self.smoothing_factor)
        prob_if_token_given_not_category = (self.smoothing_factor + self.probs[category][self.neg][token])/(self.datapoints_count - self.category_count[category] + 2 * self.smoothing_factor)
        prob_if_category_given_token = prob_if_token_given_category/ (prob_if_token_given_category + prob_if_token_given_not_category)
        return prob_if_category_given_token

    def predict(self, text) -> int:
        tokens =  dumb_tokenize(text)
        probs = np.zeros(len(self.categories))
        for i, category in enumerate(self.categories):
            temp_probs = np.zeros(len(tokens))
            for j, token in enumerate(tokens):
                temp_probs[j] = self._calculate_probabilities(token, category)
            probs[i] = np.exp(np.sum(np.log(temp_probs)))
        print(probs)
        return (self.categories[np.argmax(probs)], probs[np.argmax(probs)])

class Datapoint(NamedTuple):
    text:str
    category:str
    
def test_run():
    categories = ["spam", "ham"]
    training_examples = [Datapoint("spam rules", "spam"), Datapoint("ham rules", "ham"), Datapoint("hello ham", "ham")]
    clf = NaiveBayesClassifier(categories, smoothing_factor=0.5)
    clf.train(training_examples)
    print(clf.predict("hello spam"))

if __name__ == "__main__":
    test_run()