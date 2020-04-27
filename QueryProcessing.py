import nltk
nltk.download('punkt')
import os

# creating list of all data files under folder tf_idf_output
file_list=os.listdir("tf_idf_output/")

# List containing 5 queries
# 1. Son of God
# 2. Anarchism draws on many currents of thought and strategy
# 3. question as to whether or not
# 4. tomb of Abner
# 5. patterns of atoms larger than hydrogen
Query = ["Son of God", "Anarchism draws on many currents of thought and strategy", "question as to whether or not", "tomb of Abner", "patterns of atoms larger than hydrogen"]


for q in Query:
    query_weights = {}
    words = nltk.word_tokenize(q)
    # iterate for all data files
    for doc in file_list:
        file = open("tf_idf_output/" + doc, "r+", encoding='utf-8')
        x = file.read()
        # parsing text file to dictionary
        tfidf = eval(x)
        keys = list(tfidf.keys())
        for key in keys:
            # key[0] is document id, key[1] is word
            if key[1] in words:
                try:
                    query_weights[key[0]] += tfidf[key]
                except:
                    query_weights[key[0]] = tfidf[key]

    # sort the query weights on basis of score from higher to lower
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    print("")
    # l is list of top 10 sorted documents
    # score is list of scores associated with documents
    l = []
    score = []
    for i in query_weights[:10]:
        l.append(i[0])
        score.append(i[1])
    print("1. " + q + " : top 10 doc : " + str(l) + "and corresponding scores : " + str(score))