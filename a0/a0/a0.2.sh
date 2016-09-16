cat cmudict.tsv | sed -r "s/[^\t]*\t[^\t]*\t//" | sort | uniq -c | sort -nr | grep -vp "^   1 "
