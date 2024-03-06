from flask import Flask, request, jsonify, g
import gzip
import pandas as pd
import pickle
from collections import Counter
# from inverted_index_body_colab import *
# from inverted_index_title_colab import *
from google.cloud import storage
import logging
from sim_calc import *
import io


# Set up logging
logging.basicConfig(level=logging.DEBUG)



bucket_name = '318305570_207757535_bucket'

body_URL = 'index_posting_locs_body/index_body.pkl'
training_URL = 'index_posting_locs_training/index_body_genetics.pkl'
title_URL = 'index_posting_locs_title/index_title.pkl'
title_not_stemmed_URL = 'index_posting_locs_title_not_stemmed/index_title_not_stemmed.pkl'
# title_training_URL = 'index_posting_locs_title_training/index_title_training.pkl'
pr_URL = 'page_rank_norm.pkl'
pv_URL = 'page_view_norm.pkl'
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob_body = bucket.blob(body_URL)
# blob_training = bucket.blob(training_URL)
blob_title = bucket.blob(title_URL)
# blob_title_not_stemmed = bucket.blob(title_not_stemmed_URL)
# blob_title_training = bucket.blob(title_training_URL)
blob_pv = bucket.blob(pv_URL)
blob_pr = bucket.blob(pr_URL)
contents_body = blob_body.download_as_bytes()
# contents_title_training = blob_title_training.download_as_bytes()
# contents_training = blob_training.download_as_bytes()
contents_title = blob_title.download_as_bytes()
contents_pr = blob_pr.download_as_bytes()
contents_pv = blob_pv.download_as_bytes()
norm_pr = pickle.loads(contents_pr)
# index_title_training = pickle.loads(contents_title_training)
index_body = pickle.loads(contents_body)
# index_training = pickle.loads(contents_training)
index_title = pickle.loads(contents_title)
norm_pv = pickle.loads(contents_pv)
# max_pv_value = max(pv_dict.values())
# norm_pv = {doc_id: view/max_pv_value for doc_id, view in pv_dict.items()}
# print(f'score: {norm_pr.get(519482)}')
# print(f'page rank size: {len(norm_pr.keys())}')
# for key, value in norm_pr.items():
#     if val != norm_pr_excel_dict.get(key):
#         print('wrong')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# body_URL = storage_client.list_blobs(bucket_name, prefix='index_posting_locs_body')
# training_URL = storage_client.list_blobs(bucket_name, prefix='index_posting_locs_training')
# title_URL = storage_client.list_blobs(bucket_name, prefix='index_posting_locs_title')
# pr_URL = 'Page_Rank'
# pv_URL = 'Page_View'
# index_body_GCP = 'index_body'
# index_training_GCP = 'index_body_genetics'
# index_title_GCP = 'index_title'


'''change before GCP'''


# body_index = InvertedIndex.read_index(body_URL, index_body_GCP, bucket_name)
# training_index = InvertedIndex.read_index(training_URL, index_training_GCP, bucket_name)
# title_index = InvertedIndex.read_index(title_URL, index_title_GCP, bucket_name)

# body_index = InvertedIndex.read_index(body_URL, index_body_GCP)

# with open(pr_URL, 'rb') as pr_file:
#     df = pd.read_csv(pr_file, header=None)
#     # Extract the first and second columns, excluding the first row
#     first_column = df.iloc[0:, 0]
#     second_column = df.iloc[0:, 1]
#     # Create a dictionary from the columns
#     pr_dict = dict(zip(first_column, second_column))

# with open(pv_URL, 'rb') as pv_file:
#     pv_dict = pickle.load(pv_file)


@app.route("/search")
def search():
# @app.route("/search/<title_weight>/<cosin_weight>/<BM25_weight>/<pr_weight>/<pv_weight>")
# def search(title_weight, cosin_weight, BM25_weight, pr_weight, pv_weight):
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
    try:
        title_weight = 0.3
        cosin_weight = 0
        BM25_weight = 0.2
        pr_weight = 0.5
        pv_weight = 0
        res = []
        query = request.args.get('query', '')
        if len(query) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        '''check how to do tzerufey smichut'''
        query_tokens = sim_calc.tokenize(query)
        # print(f'query_tokens: {query_tokens}')
        sim_query_tokens = []
        for token in query_tokens:
            sim_tokens = [t[0] for t in sim_calc.find_most_similar(token)]
            sim_query_tokens.extend(sim_tokens)
            print(f'sim_query_tokens after token {token}: {sim_query_tokens}')
        sim_query_tokens.extend(query_tokens)
        # # print(f'final query tokens: {sim_query_tokens}')
        stemmed_query_tokens = sim_calc.stemming(sim_query_tokens)
        # print(f'stemmed tokens: {stemmed_query_tokens}')
        # stemmed_query_tokens = sim_calc.stemming(query_tokens)
        # print(f'stemmed tokens: {stemmed_query_tokens}')
        rank_res_cos = Counter()
        rank_res_BM25 = Counter()
        rank_res_comp = Counter()
        norm_res = Counter()
        if cosin_weight != 0 and BM25_weight == 0:
            rank_res_cos = sim_calc.cosin(stemmed_query_tokens, index_body, cosin_weight, norm_pr, norm_pv, pr_weight,
                                      pv_weight)
            max_cos_value = max(rank_res_cos.values())
            norm_res = Counter({doc_id: view / max_cos_value for doc_id, view in rank_res_cos.items()})

        if BM25_weight != 0 and cosin_weight == 0:
            print('BM25')
            rank_res_BM25 = sim_calc.BM25(stemmed_query_tokens, index_body, BM25_weight, norm_pr, norm_pv, pr_weight,
                                     pv_weight)
            max_BM25_value = max(rank_res_BM25.values())
            norm_res = Counter({doc_id: view / max_BM25_value for doc_id, view in rank_res_BM25.items()})


        if BM25_weight != 0 and cosin_weight != 0:
            rank_res_comp = sim_calc.compound_ranking(BM25_weight, cosin_weight, stemmed_query_tokens, index_training, pr_dict, pv_dict, pr_weight, pv_weight)
            max_comp_value = max(rank_res_comp.values())
            norm_res = Counter({doc_id: view / max_comp_value for doc_id, view in rank_res_comp.items()})

        if title_weight != 0:
            sum_rank_res = sim_calc.get_title_binary_score(index_title, stemmed_query_tokens, title_weight, norm_res, norm_pr,
                                                           norm_pv, pr_weight, pv_weight)
            all_doc_score_pairs_norm = [(x[0], x[1] / len(query_tokens)) for x in sum_rank_res]
            for id, score in list(all_doc_score_pairs_norm)[:100]:
                # print(index_title_training.id_title_dict[id], id, score)
                res.append((str(id), score))
        else:
            for id, score in (norm_res.most_common())[:100]:
                # print(index_title_training.id_title_dict[id], id, score)
                res.append((str(id), score))
        return jsonify(res)
    except Exception as e:
        # Log the exception
        logging.exception("An error occurred in search function: %s", e)
        # Handle the exception
        response = ('error', str(e))

        return jsonify(response)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
