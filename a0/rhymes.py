"""
ENLP A0.5-6: Given an English word, list its rhymes
based on the CMU Pronouncing Dictionary.
"""

import json, re, sys, doctest

CMUDICT = {}    # word -> pronunciations

def load_dict():
    """Load cmudict.json into the CMUDICT dict."""
    INPUT_PATH = 'cmudict.json'
    # TODO

def pronunciations(word):
    """Get the list of possible pronunciations for the given word
    (as a list of lists) by looking the word up in CMUDICT.
    If the word is not in the dictionary, return None.

    TODO: write a few doctests below.

    >>>
    """
    # TODO
    pass

def rhyming_words(word):
    """Get the list of words that have a pronunciation
    that rhymes with some pronunciation of the given word.

    >>> 'STEW' in rhyming_words('GREW')
    True

    >>> 'GROW' in rhyming_words('GREW')
    False

    >>> 'GREW' in rhyming_words('GREW')
    True

    TODO: write more doctests
    """
    # TODO
    pass

if __name__=='__main__':
    load_dict()

    doctest.testmod()   # run doctests

    query = ... # TODO: get the word provided as input

    if query not in CMUDICT:
        # TODO: print error to stderr
    else:
        for w in sorted(rhyming_words(query)):
            print(w)
