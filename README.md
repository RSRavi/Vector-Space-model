# Vector-Space-modal
Ranking of documents on basis of tf-idf score

Run file 'IndexCreation.py' creates documents data on basis of tf-idf score
1. creates term frequency dictionary
2. creates document frequency dictionary
3. finally, create a dictionary contains tf-idf score
--- create folder name "tf_idf_output"
4. finally add files on basis of similar doc_id to tf_idf_output folder e.g : doc_id[1].txt, doc_id[2].txt, doc_id[3].txt....

Run file "QueryProcessing.py" finds top 10 relevant dictionary on basis of query
1. iterate each data file to find matching query
2. return top 10 documents on basis of relevant query

Note: 

Ranking using Matching Score
=============================
Matching score is the most simplest way to calculate the similarity, in this method, we add tf_idf values of the tokens that are in query for every document. for example, if the query “hello world”, we need to check in every document if these words exists and if the word exists, then the tf_idf value is added to the matching score of that particular doc_id. in the end we will sort and take the top k documents.
