
# coding: utf-8

# In[6]:

import json


# In[11]:

d = []
with open("cmudict.tsv") as f:
        for line in f:
            an = line.strip('\n').split('\t')
            # strips the newline character from the end of each line
            d.append({'word': an[0], 'pronunciations': (an[2]).split(' ')})
            # first pronoun is whitespace so instead of an[1:], an[2:] was used


# In[3]:

### found on https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch06s02.html 
with open('cmudict.json', 'w') as f:
     json.dump(d, f)

