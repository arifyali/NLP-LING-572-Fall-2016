grep -p '^[a-zA-Z]' cmudict-0.7b | sed -r 's/\(/\t/' | sed -r 's/\)  /\t/' | sed -r 's/  /\t\t/g' > cmudict.tsv
