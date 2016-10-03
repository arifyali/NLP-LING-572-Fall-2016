#!/usr/bin/env python3
"""
ENLP A1 Part II: Perceptron

Usage: python perceptron.py NITERATIONS

(Adapted from Alan Ritter)
"""
import sys, os, glob

from collections import Counter
from math import log
from numpy import mean

from nltk.stem.wordnet import WordNetLemmatizer

from evaluation import Eval

from nbmodel import load_docs

def extract_feats(doc):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()

    ...

    return ff

def load_featurized_docs(datasplit):
    rawdocs, labels = load_docs(datasplit, lemmatize=False)
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d))
    return featdocs, labels

class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None):
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {l: Counter() for l in self.CLASSES}
        self.learn(train_docs, train_labels)

    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}

    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        ...

    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        return ...

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        return ...

    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        return ev.accuracy()


if __name__ == "__main__":
    args = sys.argv[1:]
    niters = int(args[0])

    train_docs, train_labels = load_featurized_docs('train')
    print(len(train_docs), 'training docs with',
        sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

    dev_docs,  dev_labels  = load_featurized_docs('dev')
    print(len(dev_docs), 'dev docs with',
        sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)


    test_docs,  test_labels  = load_featurized_docs('test')
    print(len(test_docs), 'test docs with',
        sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)
    acc = ptron.test_eval(test_docs, test_labels)
    print(acc, file=sys.stderr)
