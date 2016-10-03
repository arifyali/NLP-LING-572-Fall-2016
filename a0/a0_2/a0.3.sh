cut -f1 cmudict.tsv | grep -E "([^aeiouAEIOU]?[aeiouAEIOU]+[^aeiouAEIOU]?){2,}"
# a count of syllable was done by counting the number of vowels (along the with optional consonant margins) in each word and returning words with more than two. 
#I based this off of a wikipedia article that states "A syllable is typically made up of a syllable nucleus (most often a vowel) with optional initial and final margins (typically, consonants)." 
# Using this regex, the words had the following syllables:
# LIEUTENANT: 3
# TUITION: 3
# CHOIRS: 2
# FAMILY: 2
# INTERESTING: 4 
# NORMALLY: 2
# The One issue is the ending Y as somethings Y's are vowels
