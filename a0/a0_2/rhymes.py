
# coding: utf-8

# In[14]:

"""
ENLP A0.5-6: Given an English word, list its rhymes
based on the CMU Pronouncing Dictionary.
"""

import json, re, sys, doctest, argparse


# In[15]:

def load_dict():
    """Load cmudict.json into the CMUDICT dict."""
    INPUT_PATH = 'cmudict.json'
    data = []
    with open(INPUT_PATH) as f:
        for line in f:
            data.append(json.loads(line))
    data = {i['word']: i['pronunciations'] for i in data}
    #data = {item.keys()[0]: item.values[0] for item in data}
    return(data)
# figured out how to load from http://stackoverflow.com/questions/12451431/loading-and-parsing-a-json-file-in-python
CMUDICT = load_dict()


# In[16]:

def pronunciations(word):
    """Get the list of possible pronunciations for the given word
    (as a list of lists) by looking the word up in CMUDICT.
    If the word is not in the dictionary, return None.

    TODO: write a few doctests below.

    >>> pronunciations("DOG")
    [['D', 'AO1', 'G']]

    >>> pronunciations("ZYWICKI")
    [['Z', 'IH0', 'W', 'IH1', 'K', 'IY0']]

    >>> pronunciations("dog")

    """
    #word = word.upper()

    try:
        match = CMUDICT[word]
    except KeyError:
        match = None
    return(match)
            #used http://stackoverflow.com/questions/17149561/how-to-find-a-value-in-a-list-of-python-dictionaries

    pass


# In[17]:

def ipa_vowels(string):
    # used the wikipedia article https://en.wikipedia.org/wiki/Arpabet
    # and http://www.phon.ucl.ac.uk/home/wells/ipa-unicode.htm to make dict
    ipa_dict = {'AXR' : '\u025A', 'EH R' : '\u025Br', 'UH R' : '\u028Ar',
                'AO R' : '\u0254r', 'AA R' : '\u0251r', 'IY R' : '\u026Ar',
                'IH R' : '\u026Ar', 'AW R' : 'a\u028Ar',
                'AA': '\u0251', 'AO': '\u0254', 'IY': 'i', 'UW': 'u',
               'EH': '\u025B', 'IH': '\u026A', 'UH': '\u028A', 'AH1': '\u028C',
               'AH0': '\u0259', 'AX': '\u0259', 'AE': '\u00E6', 'EY': 'e\u026A',
               'AY': 'a\u026A', 'OW': 'o\u028A', 'AW': 'a\u028A', 'OY' : '\u0254\u026A',
               'ER' : '\u025A'}
    for ipa in ipa_dict.keys():
        string = string.replace(ipa, ipa_dict[ipa])
    return(string)



# In[18]:

def rhyming_words(word, flag = 0):
    """Get the list of words that have a pronunciation
    that rhymes with some pronunciation of the given word.

    >>> 'STEW' in rhyming_words('GREW')
    True

    >>> 'GROW' in rhyming_words('GREW')
    False

    >>> 'GREW' in rhyming_words('GREW')
    True

    >>> 'DOG' in rhyming_words('FOG')
    True

    >>> 'DOG' in rhyming_words('CAT')
    False

    """
    rhyme_match = []
    word_pronunciation = pronunciations(word)

    for possible_match in CMUDICT.keys(): # This goes through every word in CMUDICT
        pmpron = pronunciations(possible_match) #pronunciations for each word
        for pron in pmpron: # if a word has multiple spellings
            for i in word_pronunciation: # the word supplied, this for loop checks the multiple pronunciations case
                rhyme_check = i[-1]
                if pron[-1] == rhyme_check: #logic is that if the last syl matches for the target word and the candidate, the candidate is a rhyming word
                    if flag == 0: #determines if output is list of words or not
                        rhyme_match.append(possible_match)
                    else:
                        pron_string = ' '.join(pron)
                        print(ipa_vowels(pron_string))

    return(rhyme_match)
    pass


# In[6]:

if __name__=='__main__':
    load_dict()

    doctest.testmod()   # run doctests

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="output pronunciations, not words",
                        action="store_true")
    parser.add_argument("q", type=str)

    args = parser.parse_args()

    query = (args.q).upper()

    if query not in CMUDICT:
        sys.stderr.write("Query word not found")
    else:
        if args.p:
            print("test")
            for w in sorted(rhyming_words(query, flag =1)):
                print(w)
        else:
            for w in sorted(rhyming_words(query)):
                print(w)

