#!/usr/bin/env python3
"""
ENLP A1 Part I: Naive Bayes

(Adapted from Alan Ritter)
"""
import sys, os, glob

import numpy as np
from collections import Counter
from math import log

from nltk.stem.wordnet import WordNetLemmatizer

from evaluation import Eval

def load_docs(direc, lemmatize, labelMapFile='labels.csv'):
    """Return a list of word-token-lists, one per document.
    Words are optionally lemmatized with WordNet."""


    labelMap = {}   # docID => gold label, loaded from mapping file
    with open(os.path.join(direc, labelMapFile)) as inF:
        for ln in inF:
            docid, label = ln.strip().split(',')
            assert docid not in labelMap
            labelMap[docid] = label

    # create parallel lists of documents and labels
    docs, labels = [], []
    for file_path in glob.glob(os.path.join(direc, '*.txt')):
        filename = os.path.basename(file_path)
        labels.append(labelMap[filename])
        with open(file_path) as f:
            docs.append(f.read().split())
            #if lemmatize:
            #    for word in range(0,doc):

        # credit: http://stackoverflow.com/questions/13259288/returning-a-list-of-words-after-reading-a-file-in-python
        ...

    return docs, labels

class NaiveBayes:
    def __init__(self, train_docs, train_labels, ALPHA=1.0):
        # list of native language codes in the corpus
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']

        self.ALPHA=ALPHA
        self.priorProbs = {l: 0 for l in self.CLASSES}
        self.likelihoodProbs = {l: Counter() for l in self.CLASSES}
        self.trainVocab = set()
        self.learn(train_docs, train_labels, alpha=self.ALPHA)

    def learn(self, docs, labels, alpha=1.0):
        """Estimate parameters for a naive Bayes bag-of-words model with the
        given training data and amount of add-alpha smoothing."""

        assert len(docs)==len(labels)
        labelCounts = {l: 0 for l in self.CLASSES}
        wordCounts = {l: Counter() for l in self.CLASSES}
        totalWordCounts = {l: 0 for l in self.CLASSES}
        # iterate over documents in order to record

        for i in range(0, len(labels)):
        # count(y) in labelCounts
            l = labels[i]
            labelCounts[labels[i]] +=1
            # count(y,w) for all words in totalWordCounts
            totalWordCounts[labels[i]] += len(docs[i])
            words = docs[i]
            # count(y,word) in wordCounts,
            for word in words:
                wordCounts[labels[i]][word] += 1
                # and to store the training vocabulary in self.trainVocab
                self.trainVocab.add(word)
        # compute and store prior distribution over classes
        # (unsmoothed) in self.priorProbs
            for word in self.trainVocab: 
                self.likelihoodProbs[l][word] = np.divide(wordCounts[l][word]+self.ALPHA, totalWordCounts[l]+self.ALPHA*(len(self.trainVocab)+1))
            self.likelihoodProbs[l]['**OOV**'] = np.divide(self.ALPHA, totalWordCounts[l]+self.ALPHA*(len(self.trainVocab)+1))

        # Sanity checks--do not modify
        assert len(self.priorProbs)==len(self.likelihoodProbs)==len(self.CLASSES)>2
        assert .999 < sum(self.priorProbs.values()) < 1.001
        for y in self.CLASSES:
            assert .999 < sum(self.likelihoodProbs[y].values()) < 1.001,sum(self.likelihoodProbs[y].values())
            assert 0 <= self.likelihoodProbs[y]['**OOV**'] < 1.0,self.likelihoodProbs[y]['**OOV**']

    def joint_prob(self, doc, y):
        joint = log(self.priorProbs[y])
        # compute the log of the joint probability of the document and the class,
        # i.e., return p(y)*p(w1|y)*p(w2|y)*... (but in log domain)
        for word in doc:
            joint += log(self.likelihoodProbs[y][word])
                
        # should not make any changes to the model parameters
        return joint

    def predict(self, doc):
        max_joint = 0
        max_class = 'unclassified'
        for l in self.CLASSES:
            joint_prob = self.joint_prob(doc, l)
            if joint_prob>max_joint:
                max_joint = joint_prob
                max_class = l
        # apply Bayes' rule: return the class that maximizes the
        # prior * likelihood probability of the test document
        # should not make any changes to the model parameters
        return max_class

    def eval(self, test_docs, test_labels):
        """Evaluates performance on the given evaluation data."""
        assert len(test_docs)==len(test_labels)
        preds = []  # predicted labels
        for doc,y_gold in zip(test_docs,test_labels):
            y_pred = self.predict(doc)
            preds.append(y_pred)
        ev = Eval(test_labels, preds)
        return ev.accuracy()

if __name__ == "__main__":
    args = sys.argv[1:]
    lemmatize = False
    if args[0]=='-l':
        lemmatize = True
        args = args[1:]
    alpha = float(args[0])

    train_docs, train_labels = load_docs('train', lemmatize)
    print(len(train_docs), 'training docs with',
        sum(len(d) for d in train_docs), 'tokens', file=sys.stderr)

    test_docs,  test_labels  = load_docs('dev', lemmatize)
    print(len(test_docs), 'eval docs with',
        sum(len(d) for d in test_docs), 'tokens', file=sys.stderr)

    nb = NaiveBayes(train_docs, train_labels, alpha)

    print(nb.eval(test_docs, test_labels))
