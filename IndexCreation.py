from bs4 import BeautifulSoup
import nltk
import math
nltk.download('punkt')
import re
from nltk.util import ngrams
from nltk import FreqDist

# read file using beautifulsoup
soup = BeautifulSoup(open('wiki_00',encoding='utf-8'),"html.parser")
text = soup.get_text()
# seperate corpus on tag <doc>
doc = soup.find_all("doc")
doc_id=[]
structure = {}
e={}
tf = {}
idf = {}

# creating list of documents id
for id in soup.find_all('doc'):
    doc_id.append(id.get('id'))

# text pre-processing(removing white space, lowering alphabets)
def clear_text(text):
    text = text.lower()
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\s+',' ',text)
    words = nltk.word_tokenize(text)
    return words

# method to generate n-grams
def get_ngram(tokens, n):
    n_gram = ngrams(tokens,n)
    return [' '.join(grams) for grams in n_gram]

words=clear_text(text)
unigram = get_ngram(words,1)
unique = FreqDist(unigram)

# create dictionary of document wrt its associated tokenized words
# tf --> {(<doc_id>, <word>): <number of times word repeated in document>}
# structure --> inverse index of words : {<doc_id>: <tokenized_words>}
def term_frequency():
    for id in doc_id:
        print("start tf : ", id)
        # seperate corpus on basis of id
        text=soup.find(id=id).get_text()
        # basic pre-processing using clear_text method
        words = clear_text(text)
        unigram = get_ngram(words,1)
        doc_token = FreqDist(unigram)
        structure[id]=doc_token.keys()
        for x in list(unique.keys()):
            if x in list(doc_token.keys()):
                tf[id, x]=doc_token[x]/len(doc_token)
            else:
                tf[id, x]=0
    return tf,structure

# creating inverse index of documents
def inv_index():
    x, structure=term_frequency()
    for x in unique.keys():
        c=[]
        for i in doc_id:
            if x in structure[i]:
                c.append(i)
        e[x]=len(c)
    return e

# creating dictionary for document frequency
# df --> {(<doc_id>, <word>): <number of document containing that word>}
def doc_frequency():
    e = inv_index()
    print("start df : ")
    for x in unique.keys():
        for i in doc_id:
            idf[i, x] = e[x]
    return idf

# finally calculating tf-idf score using method doc_frequency() and term_frequency()
# return tf-idf dictionary on basis of score --> {(<doc_id>, <word>): tf*idf}
# finally create files on basis of similar doc_id e.g : doc_id[1].txt, doc_id[2].txt, doc_id[3].txt....
def calc_tf_idf():
    print("start tf-idf : ", id)
    tf,n = term_frequency()
    idf = doc_frequency()
    ###########################
    # tf-idf = tf*ln(n/df)    #
    # tf = term_frequency()   #
    # df = doc_frequency()    #
    ###########################
    res = {k: tf[k]*math.log(len(doc_id)/idf[k], 10) for k in tf}
    for i in doc_id:
        n = []
        nd = {}
        for x in res:
            if x[0] == i:
                n.append(x)
        for x in n:
            nd[x] = res[x]
        print(nd)
        filename = "tf_idf_output/"+i+".txt"
        f = open(filename, "w", encoding='utf-8')
        f.write(str(nd))
        f.close()

calc_tf_idf()