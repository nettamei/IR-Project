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
- Building inverted indexs (title, body, anchor, page rank, page view)

In order to reduce search time, we added a few data structures to the inverted indexes during the preprocessing of the data:


Query Preprocessing
In addition to the preprocessing we did on the query, we gave weight to each token in the query.
Words that were inside quotation marks are given more weight because they are more likely to be the essence of the query.

Search
For the search we used 5 components:
- Title - binary ranking, does the wanted word exist in the title or not.
- BM25 - similarity measue using the formula of BM25 and B = 0.75, K = 1.2
- Anchore text - the number of times a page is referenced from anchor text
- Page view - number of times the page was viewed
- Page rank - page authority

Each of the components in the search process got different weight for the final calculation of the similarity.
The queries are seperated into 3 groups, question queries, queries of one word and others.
Each of these groups got different weights for the components, according to the relevence of each component in the specific type of query.
