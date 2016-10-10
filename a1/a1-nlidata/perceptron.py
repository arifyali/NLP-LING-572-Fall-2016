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
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('wordnet')
lm = WordNetLemmatizer()

from evaluation import Eval

from nbmodel import load_docs

from sklearn.metrics import confusion_matrix

def extract_feats(doc, uppercase = False, ngram = False, n = 1):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    ff = Counter()
    if ngram:
        for i in range((len(doc)-n+1)):
            ngram = ' '.join(doc[i:i+n])
            if uppercase:
                ngram = ngram.upper()
                 #this does the ngram 
            ngram = ' '.join(doc[i:i+n])
            ff[ngram] = 1
    else:    
        for word in doc:
            if uppercase:
                word = word.upper()
            ff[word] = 1
    ff['***bias_term***'] = 1
    return ff

def load_featurized_docs(datasplit, uppercase = False, lemmatize = False, ngram = False, n = 1):
    rawdocs, labels = load_docs(datasplit, lemmatize)
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d, uppercase, ngram, n))
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
        #print("iteration,train_accuracy,dev_accuracy,update", file=sys.stderr)
        for iteration in range(self.MAX_ITERATIONS):
            update = 0
            train_accuracy = 0
            for i in range(len(train_docs)):
                label = train_labels[i]
                yhat = self.predict(train_docs[i])
                if yhat != label:
                    for word in train_docs[i]:
                        self.weights[label][word] += train_docs[i][word]
                        self.weights[yhat][word] -= train_docs[i][word]
                    train_accuracy -= 1
                    update += 1
            #print(str(iteration) +","+ str(np.divide(len(train_docs)+train_accuracy, len(train_docs))) +"," + str(self.test_eval(dev_docs, dev_labels))+","+str(update), file=sys.stderr)
            if np.divide(len(train_docs)+train_accuracy, len(train_docs)) == 1.0:
                break
                                                

    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        score = 0
        for word in doc:
            score += self.weights[label][word]*doc[word]
        return score

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        max_label = self.CLASSES[0]
        max_score = self.score(doc, max_label)
        # note: the dict method from nbmodels don't work because there is a greater certainity of 
        for l in self.CLASSES[1:]:
            current_score = self.score(doc, l)
            if current_score > max_score:
                max_score = current_score
                max_label = l
        return max_label


    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        cm = confusion_matrix(test_labels, pred_labels, labels = self.CLASSES)
        # I used the sklearn package, which makes building confusion matices similar to R
        print(cm, file=sys.stderr)
        for l in range(len(self.CLASSES)):
            tp = cm[l][l]
            tp_fn = sum(cm[l])
            tp_fp = 0
            for i in range(len(self.CLASSES)): # i can't select column from array so I have to loop
                tp_fp += cm[i][l]
            precision = np.divide(tp,tp_fp)
            recall = np.divide(tp,tp_fn)
            f1 = np.divide(2*precision*recall, precision+recall)
            print("precision for " + l +": ", +str(precision), file=sys.stderr)
            print("recall for " + l +": ", +str(recall), file=sys.stderr)
            print("F1 for " + l +": ", +str(f1), file=sys.stderr)
            label_weights = self.weights[l]
            print("max weights for " + l + ": " + str(sorted(label_weights, key=label_weights.get, reverse = True)[:10]), file=sys.stderr)
            print("min weights for " + l + ": " + str(sorted(label_weights, key=label_weights.get)[:10]), file=sys.stderr)
            print("bias feature for "+ l + ": " + str(label_weights['***bias_term***']),file=sys.stderr)

     
    #    ev = Eval(test_labels, pred_labels)
    #    return ev.accuracy()


if __name__ == "__main__":

    args = sys.argv[1:]

    average = False
    if args[0] == '-l':
        lemmatize = True
        args = args[1:]
    else:
        lemmatize = False

    if args[0] == '-u':
        uppercase = True
        args = args[1:]
    else:
        uppercase = False

    
    if args[0] == '-n':
        ngram = True
        n = int(args[1])
        args = args[2:]
    else:
        ngram = False
        n = 1

    # work later
    if args[0] == '-a':
        average = True
    else:
        average = False

    niters = int(args[0])

    train_docs, train_labels = load_featurized_docs('train',uppercase, lemmatize, ngram, n)
    #print(len(train_docs), 'training docs with',
    #    sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

    dev_docs,  dev_labels  = load_featurized_docs('dev', uppercase, lemmatize, ngram, n)
    #print(len(dev_docs), 'dev docs with',
    #    sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)


    test_docs,  test_labels  = load_featurized_docs('test',uppercase, lemmatize, ngram, n)
    #print(len(test_docs), 'test docs with',
    #    sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)
    acc = ptron.test_eval(test_docs, test_labels)
    print(acc, file=sys.stderr)
