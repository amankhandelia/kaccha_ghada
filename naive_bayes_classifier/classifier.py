from typing import List, NamedTuple
from tokenizer import dumb_tokenize

# https://stats.stackexchange.com/questions/142505/how-to-use-naive-bayes-for-multi-class-problems
class NaiveBayesClassifier:
    def __init__(self, categories):
        self.probs = {}
        self.categories = categories
        self.category_count = {category:0 for category in self.categories}
        pass

    def train(self, dataset:List[NamedTuple]) -> None:
        for datapoint in dataset:
            self.category_count[datapoint.category] += 1
            for token in dumb_tokenize(datapoint.text):
                self.probs[datapoint.category][self.pos][token] += 1
                for category in self.categories:
                    if category != datapoint.category:
                        self.probs[datapoint.category][self.neg][token] += 1

        pass

    def _calculate_probabilities(self) -> float:
        pass

    def predict(self, text) -> int:
        pass