from typing import List, NamedTuple, Set, Dict, Tuple
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
        self.probs:Dict[str, Dict[str, Dict[str, int]]] = {}
        self.categories:List = categories
        self.category_count:Dict[str, int] = {category:0 for category in self.categories}
        self.datapoints_count = 0
        self.pos = 'pos'
        self.neg = 'neg'
        self.smoothing_factor = smoothing_factor
        self.tokens:Set = set()
        self._initialize_counters()
        
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
                self.tokens.add(token)
                self.probs[datapoint.category][self.pos][token] += 1
                for category in self.categories:
                    if category != datapoint.category:
                        self.probs[category][self.neg][token] += 1
        return

    def _calculate_probabilities(self, token:str, category:str) -> Tuple[float, float]:
        prob_if_token_given_category = (self.smoothing_factor + self.probs[category][self.pos][token])/(self.category_count[category] + 2 * self.smoothing_factor)
        prob_if_token_given_not_category = (self.smoothing_factor + self.probs[category][self.neg][token])/(self.datapoints_count - self.category_count[category] + 2 * self.smoothing_factor)
        return prob_if_token_given_category, prob_if_token_given_not_category

    def predict(self, text) -> int:
        tokens =  dumb_tokenize(text)
        probs = np.zeros(len(self.categories))
        for i, category in enumerate(self.categories):
            temp_probs_category = np.zeros(len(self.tokens))
            temp_probs_not_category = np.zeros(len(self.tokens))
            for j, token in enumerate(self.tokens):
                if token in tokens:
                    temp_probs_category[j], temp_probs_not_category[j] = self._calculate_probabilities(token, category)
                else:
                    # probs of not seeing the token given the category
                    temp_probs_category[j], temp_probs_not_category[j] = self._calculate_probabilities(token, category)
                    temp_probs_category[j], temp_probs_not_category[j] = 1. - temp_probs_category[j], 1. - temp_probs_not_category[j]
            probs[i] = np.exp(np.sum(np.log(temp_probs_category)))/(np.exp(np.sum(np.log(temp_probs_category))) + np.exp(np.sum(np.log(temp_probs_not_category))))
        return (self.categories[np.argmax(probs)], probs[np.argmax(probs)])

class Datapoint(NamedTuple):
    text:str
    category:str
    
def test_run():
    categories = ["spam", "ham"]
    training_examples = [Datapoint("spam rules", "spam"), Datapoint("ham rules", "ham"), Datapoint("hello ham", "ham")]
    clf = NaiveBayesClassifier(categories, smoothing_factor=0.5)
    clf.train(training_examples)
    assert clf.predict("hello spam")[1] > 0.82 and clf.predict("hello spam")[1] < 0.84

if __name__ == "__main__":
    test_run()