
# coding: utf-8

# In[1]:

import json, collections


# In[2]:

s = []
with open("cmudict.tsv") as f:
    for line in f:
        an = line.strip('\n').split('\t')
        #strips the newline character
        s.append((an[0], (an[2]).split(' ')))
        #an[1] is white space


# In[3]:

d = collections.defaultdict(list)
for k, v in s:
    d[k].append(v)


# In[4]:

for k, v in d.items():
    data = {"word": k, "pronunciations": v}
    with open('cmudict.json', 'a') as outfile:
        json.dump(data, outfile)
        outfile.write('\n')

