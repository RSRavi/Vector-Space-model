from bs4 import BeautifulSoup
import nltk
import math
nltk.download('punkt')
import re

# read file using beautifulsoup
soup = BeautifulSoup(open('wiki_00',encoding='utf-8'),"html.parser")
text = soup.get_text()
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
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
text = text.lower()
text = re.sub(r'\W',' ',text)
text = re.sub(r'\s+',' ',text)
words = nltk.word_tokenize(text)
unique=[]

# create list of all unique words in corpus
for letter in words:
    if letter not in unique:
        unique.append(letter)

# create dictionary of document wrt its associated tokenized words
# structure --> {<doc_id>: <list of tokenized word in particular doc>}
def doc_structure():
    for id in doc_id:
        # seperate corpus on basis of id
        text=soup.find(id=id).get_text()
        # basic pre-processing
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        tokenized_words =nltk.word_tokenize(text)
        structure[id] = tokenized_words
    return structure

# creating inverse index of documents
def inv_index():
    for x in unique:
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
    for x in unique:
        for i in doc_id:
            idf[i, x] = e[x]
    return idf

# creating dictionary for term frequency
# tf --> {(<doc_id>, <word>): <number of word containing a document>/length of tokens in document}
def term_frequency():
    structure = doc_structure()
    for i in doc_id:
        for x in unique:
            if x in structure[i]:
                tf[i, x] = structure[i].count(x)/len(structure[i])
            else:
                tf[i, x] = 0
    return tf

# finally calculating tf-idf score using method doc_frequency() and term_frequency()
# return tf-idf dictionary on basis of score --> {(<doc_id>, <word>): tf*idf}
# finally create files on basis of similar doc_id e.g : doc_id[1].txt, doc_id[2].txt, doc_id[3].txt....
def calc_tf_idf():
    tf = term_frequency()
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