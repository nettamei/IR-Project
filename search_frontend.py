from flask import Flask, request, jsonify, g
import pickle
from google.cloud import storage
from sim_calc import *

'''Uploading the indexes'''

bucket_name = '318305570_207757535_bucket'
anchor_URL = 'index_posting_locs_anchor_text/index_anchor_text.pkl'
body_URL = 'index_posting_locs_body/index_body.pkl'
title_URL = 'index_posting_locs_title/index_title.pkl'
title_not_stemmed_URL = 'index_posting_locs_title_not_stemmed/index_title_not_stemmed.pkl'
pr_URL = 'page_rank_norm.pkl'
pv_URL = 'page_view_norm.pkl'
body_sorted_posting_list_URL = 'index_posting_locs_body_sorted_posting_list/index_body_sorted_posting_list.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob_anchor = bucket.blob(anchor_URL)
blob_body = bucket.blob(body_URL)
blob_body_sorted_posting_list = bucket.blob(body_sorted_posting_list_URL)
# blob_title = bucket.blob(title_URL)
blob_title_not_stemmed = bucket.blob(title_not_stemmed_URL)
blob_pv = bucket.blob(pv_URL)
blob_pr = bucket.blob(pr_URL)
contents_anchor = blob_anchor.download_as_bytes()
contents_body = blob_body.download_as_bytes()
contents_body_sorted_posting_list = blob_body_sorted_posting_list.download_as_bytes()
contents_title_not_stemmed = blob_title_not_stemmed.download_as_bytes()
# contents_title = blob_title.download_as_bytes()
contents_pr = blob_pr.download_as_bytes()
contents_pv = blob_pv.download_as_bytes()
norm_pr = pickle.loads(contents_pr)
index_anchor = pickle.loads(contents_anchor)
index_body = pickle.loads(contents_body)
# index_title = pickle.loads(contents_title)
index_title_not_stemmed = pickle.loads(contents_title_not_stemmed)
norm_pv = pickle.loads(contents_pv)
index_body_sorted_posting_list = pickle.loads(contents_body_sorted_posting_list)


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    # Query tokenization
    query_tokens = sim_calc.tokenize(query)

    # different component weights for different types of queries
    if len(query_tokens) == 0:
        return jsonify(res)
    elif len(query_tokens) == 1:
        title_weight = 1.5
        anchor_weight = 0.8
        cosin_weight = 0
        BM25_weight = 0.6
        pr_weight = 1
        pv_weight = 1.2
    elif '?' in query:
        title_weight = 0.7
        anchor_weight = 0.6
        cosin_weight = 0
        BM25_weight = 0.3
        pr_weight = 1
        pv_weight = 1
    else:
        title_weight = 0.7
        anchor_weight = 0.7
        cosin_weight = 0
        BM25_weight = 0.95
        pr_weight = 1
        pv_weight = 1.3


    # BEGIN SOLUTION
    stemmed_query_tokens = None
    synomyns = False
    if synomyns:
        #When building sim_query_tokens, ensure uniqueness
        sim_query_tokens = []
        for token in query_tokens:
            sim_tokens = sim_calc.get_synonyms(token[0])
            sim_query_tokens.extend(sim_tokens)
        # Filter out duplicates after stemming
        sim_query_tokens.extend(query_tokens)
        stemmed_query_tokens = sim_calc.stemming(sim_query_tokens)

    else:
        stemmed_query_tokens = sim_calc.stemming(query_tokens)

    cosin_res_id_score = Counter()
    bm25_id_score = Counter()
    comp_id_score = Counter()
    anchor_id_score = Counter()
    title_id_score = Counter()
    res_counter = Counter()

    # Similarity calculating, taking the 400 best scores from each component

    # Cosine similarity

    if cosin_weight != 0 and BM25_weight == 0:
        rank_res_cos = sim_calc.cosin(stemmed_query_tokens, index_body, cosin_weight)[:400]
        if len(stemmed_query_tokens) == 1:
            cosin_res_id_score = [(doc_id, score) for doc_id, score in rank_res_cos]
        else:
            # Normalize scores with the max score
            max_cos_value = rank_res_cos[0][1]
            cosin_res_id_score = [(doc_id, score / max_cos_value) for doc_id, score in rank_res_cos]

    # BM25 similarity
    if BM25_weight != 0 and cosin_weight == 0:
        rank_res_BM25 = sim_calc.BM25(stemmed_query_tokens, index_body, index_body_sorted_posting_list, BM25_weight)[:400]
        if len(stemmed_query_tokens) == 1:
            bm25_id_score = [(doc_id, score) for doc_id, score in rank_res_BM25]
        else:
            # Normalize scores with the max score
            max_BM25_value = rank_res_BM25[0][1]
            bm25_id_score = [(doc_id, score / max_BM25_value) for doc_id, score in rank_res_BM25]

    # Compound similarity
    if BM25_weight != 0 and cosin_weight != 0:
        rank_res_comp = sim_calc.compound_ranking(BM25_weight, cosin_weight, stemmed_query_tokens, index_body, norm_pr,
                                                  norm_pv, pr_weight)[:400]
        if len(stemmed_query_tokens) == 1:
            comp_id_score = [(doc_id, score) for doc_id, score in rank_res_comp]
        else:
            # Normalize scores with the max score
            max_comp_value = rank_res_comp[0][1]
            comp_id_score = [(doc_id, score / max_comp_value) for doc_id, score in rank_res_comp]

    # Anchor text
    if anchor_weight != 0:
        rank_res_anchor = sim_calc.anchor(index_anchor, query_tokens, anchor_weight)[:400]
        if len(stemmed_query_tokens) == 1:
            anchor_id_score = [(doc_id, score) for doc_id, score in rank_res_anchor]
        else:
            # Normalize scores with the max score
            max_anchor_value = rank_res_anchor[0][1]
            anchor_id_score = [(doc_id, score / max_anchor_value) for doc_id, score in rank_res_anchor]

    # Title
    if title_weight != 0:
        rank_res_title = sim_calc.get_title_binary_score(index_title_not_stemmed, sim_query_tokens, title_weight)[:400]
        title_id_score = [(doc_id, score) for doc_id, score in rank_res_title]

    # summing up all the scores of the components for each document

    if cosin_res_id_score:
        for doc_id, score in cosin_res_id_score:
            res_counter[doc_id] += score

    if bm25_id_score:
        for doc_id, score in bm25_id_score:
            res_counter[doc_id] += score

    if comp_id_score:
        for doc_id, score in comp_id_score:
            res_counter[doc_id] += score

    if anchor_id_score:
        for doc_id, score in anchor_id_score:
            res_counter[doc_id] += score

    if title_id_score:
        for doc_id, score in title_id_score:
            res_counter[doc_id] += score

    for doc_id in res_counter:
        try:
            res_counter[doc_id] += pv_weight * norm_pv[doc_id]
        except:
            pass

    for doc_id in res_counter:
        try:
            res_counter[doc_id] += pr_weight * norm_pr[doc_id]
        except:
            pass

    # Taking the top 100 documents

    doc_score = res_counter.most_common()
    final_doc_score = doc_score[:100]

    for i in range(0, 100):
        res.append((str(final_doc_score[i][0]), (index_title_not_stemmed.id_title_dict[final_doc_score[i][0]])[0]))

    return jsonify(res)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
