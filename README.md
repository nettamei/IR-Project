IR-Project
Ido Dai and Netta Meiri

In this project we created a search engine for 6.5 million documents of the wikipedia corpus.
In the process of building the engine, we experimented different approaches and examined the results.
The final approach used consists of:

Preprocessing the data
- Parsing
- Tokenization
- Removing stopwords
- Stemming
- Building inverted indexes (title, body, anchor, page rank, page view)

In order to reduce search time, we added a few data structures to the inverted indexes during the preprocessing of the data:
- doc_tf_idf_doc_len: <Document id: (tf-idf values, document length)> In the cosine similarity calculation we tried, the denomenator part can be precalculated because it doesn't have a query factor, only document factor.
- doc_count - number of documents in the corpus
- doc_len_avg - average length of document in the corpus
- id_title_dict: <Document id: (title, title length)> In the title index, this is for returning the final solution with the title and for normalization using title length.

Query Preprocessing
In addition to the preprocessing we did on the query, we gave weight to each token in the query.
Words that were inside quotation marks are given more weight because they are more likely to be the essence of the query.

Query Expansion
For query expansion we used the package nltk.wordnet in order to get synonyms and hyponyms for the query words and added them to the query.

Search
For the search we used 5 components:
- Title - binary ranking, does the wanted word exist in the title or not.
- BM25 - similarity measue using the formula of BM25 and B = 0.75, K = 1.2
- Anchore text - the number of times a page is referenced from anchor text
  * In order to get more useful information we modified the anchor text structure to a form that we could retrieve information about the anchor text using a token as a key and not document id (Posting list modifications).
- Page view - number of times the page was viewed
- Page rank - page authority

Inverted indexes used:

Body Inverted Index (stemmed).
Title Inverted Index (not stemmed).
Anchor text Inverted Index (stemmed).
PageRank dictionary (normalized).
PageView dictionary (normalized).

Each of the components in the search process got different weight for the final calculation of the similarity.
The queries are seperated into 3 groups, question queries, queries of one word and others.
Each of these groups got different weights for the components, according to the relevence of each component in the specific type of query.
