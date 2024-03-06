import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
import math
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# from inverted_index_body_gcp import *
# from inverted_index_title_gcp import *

bucket_name = '318305570_207757535_bucket'
PROJECT_ID = 'irfinalproject-415108'
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)


# bucket = storage.Client(PROJECT_ID).bucket(bucket_name)

class sim_calc:

    @staticmethod
    def tokenize(text):
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
        return tokens

    @staticmethod
    def stemming(tokens):
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(x) for x in tokens]
        return stemmed_tokens

    @staticmethod
    def get_synonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                # Get the normalized form of the word from the lemma
                normalized_word = lemma.name()
                # Check if the word contains any signs
                if normalized_word.isalpha():
                    normalized_word = normalized_word.lower()
                    if normalized_word:
                        synonyms.append(normalized_word)
        return synonyms

    # Define a function to find hyponyms and hypernyms
    def find_hyponyms_and_hypernyms(word):
        hyponyms = []
        synsets = wordnet.synsets(word)
        for synset in synsets:
            try:
                first_hyponym = synset.hyponyms()[0].name()
                hyponym_synset = wordnet.synset(first_hyponym)
                hypernyms = hyponym_synset.hypernyms()

                hypernym_names = [hypernym.name().split(".")[0] for hypernym in hypernyms]
                first_hypernym_name = hypernym_names[0].lower()

                if first_hypernym_name != word:
                    hyponyms.append(first_hypernym_name)
            except IndexError:
                pass  # No hyponyms found for the synset

        return hyponyms

    @staticmethod
    def find_most_similar(word, n=5):
        hyponyms_hypernyms = sim_calc.find_hyponyms_and_hypernyms(word)
        similarities = [(hyponym, wordnet.wup_similarity(wordnet.synsets(word)[0], wordnet.synsets(hyponym)[0]))
                        for hyponym in
                        hyponyms_hypernyms if
                        hyponym != word and hyponym not in all_stopwords and
                        wordnet.wup_similarity(wordnet.synsets(word)[0],
                                               wordnet.synsets(hyponym)[0]) is not None and
                        wordnet.wup_similarity(wordnet.synsets(word)[0], wordnet.synsets(hyponym)[0]) > 0.8]
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Remove duplicates
        unique_similarities = []
        seen = set()
        for sim in similarities:
            if sim[0] not in seen:
                unique_similarities.append(sim)
                seen.add(sim[0])
        return unique_similarities[:n]

    # @staticmethod
    # def find_most_similar(word, n=5):
    #     synonyms = sim_calc.get_synonyms(word)
    #     similarities = [(syn, wordnet.wup_similarity(wordnet.synsets(word)[0], wordnet.synsets(syn)[0]))
    #                     for syn in
    #                     synonyms if
    #                     syn != word and syn not in all_stopwords and wordnet.wup_similarity(wordnet.synsets(word)[0],
    #                                                                                         wordnet.synsets(syn)[
    #                                                                                             0]) is not None and
    #                     wordnet.wup_similarity(wordnet.synsets(word)[0], wordnet.synsets(syn)[0]) > 0.8]
    #     similarities.sort(key=lambda x: x[1], reverse=True)
    #     # Remove duplicates
    #     unique_similarities = []
    #     seen = set()
    #     for sim in similarities:
    #         if sim[0] not in seen:
    #             unique_similarities.append(sim)
    #             seen.add(sim[0])
    #     return unique_similarities[:n]

    @staticmethod
    def BM25(tokens, index, weight, pr_dict, pv_dict, w_pr, w_pv, K=1.2, B=0.75):
        # print('in BM25')
        query_counter = Counter(tokens)
        BM25_value_per_doc = Counter()
        doc_BM25_value = Counter()
        for token in tokens:
            # calc idf for specific token
            if index.df.get(token) is not None:
                token_df = index.df[token]
            else:
                continue
            token_idf = math.log((index.doc_count + 1) / token_df, 10)
            norm_query_token = query_counter[token] / len(tokens)
            # NEED TO CHANGE BEFORE RUNNING IN GCP
            tf_list = index.read_a_posting_list('.', token, bucket_name)
            for id, tf in tf_list:
                # if id == 12266 or id == 5717429:
                #     print(f'tf {id}: {tf}')
                #     print(f'doc length {id}: {index.doc_tf_idf_doc_len[id][1]}')
                #     print(f'normalization tf {id}: {math.log(1 + tf, 10)}')
                # normalized tf (by the length of document)
                # norm_tf = tf / index.doc_tf_idf_doc_len[id][1]
                # norm_tf = math.log(1 + tf, 10)
                norm_tf = tf
                numerator = norm_tf * (K + 1)
                # if id == 12266 or id == 5717429:
                #     print(f'numerator {id}: {numerator}')
                denominator = norm_tf + K * (1 - B + ((B * index.doc_tf_idf_doc_len[id][1]) / index.doc_len_avg))
                # if id == 12266 or id == 5717429:
                #     print(f'denominator {id}: {denominator}')
                query_factor = ((K + 1) * norm_query_token) / (K + norm_query_token)
                # if id == 12266 or id == 5717429:
                #     print(f'query_factor {id}: {query_factor}')
                if pv_dict.get(id) is None:
                    pv_dict[id] = 0
                doc_BM25_value[id] += (
                            (weight * (token_idf * (numerator / denominator) * query_factor)) + (pr_dict[id] * w_pr) + (
                                pv_dict[id] * w_pv))
        return doc_BM25_value

    @staticmethod
    def cosin(tokens, index, w_cosine, pr_dict, pv_dict, w_pr, w_pv):
        query_counter = Counter(tokens)
        numerator_value_per_doc = Counter()
        denominator_query = 0
        for token in tokens:
            if index.df.get(token) is not None:
                token_df = index.df[token]
            else:
                continue
            token_idf = math.log(index.doc_count / token_df, 10)
            norm_query_token = query_counter[token] / len(tokens)
            denominator_query += math.pow(norm_query_token * token_idf, 2)
            # NEED TO CHANGE BEFORE RUNNING IN GCP
            tf_list = index.read_a_posting_list('.', token, bucket_name)
            for id, tf in tf_list:
                norm_tf = tf / index.doc_tf_idf_doc_len[id][1]
                # optimization check add to the posting list in the end (0.5,tf*idf[token])
                numerator_value_per_doc[id] += (norm_tf * token_idf) * (norm_query_token * token_idf)
        cosin_doc_val = Counter()
        for id in numerator_value_per_doc.keys():
            if pv_dict.get(id) is None:
                pv_dict[id] = 0
            denominator_doc = index.doc_tf_idf_doc_len[id][0] / index.doc_tf_idf_doc_len[id][1]
            cosin_doc_val[id] = (
                        (w_cosine * (numerator_value_per_doc[id] / (math.sqrt(denominator_query) * denominator_doc))) +
                        (pr_dict[id] * w_pr) + (pv_dict[id] * w_pv))
        return cosin_doc_val

    @staticmethod
    def compound_ranking(alpha, beta, tokens, index, pr_dict, pv_dict, w_pr, w_pv):
        bm25_result = sim_calc.BM25(tokens, index, beta, pr_dict, pv_dict, w_pr, w_pv)
        cosin_result = sim_calc.cosin(tokens, index, alpha, pr_dict, pv_dict, w_pr, w_pv)
        compound_counter = Counter()
        for id in cosin_result.keys():
            compound_counter[id] = (alpha * bm25_result[id]) + (beta * cosin_result[id])
        return compound_counter

    @staticmethod
    def get_title_binary_score(index, tokens, weight, total_ranking_score, pr_dict, pv_dict, w_pr, w_pv):
        # print('in title')
        for token in tokens:
            # print(f'token: {token}')
            # NEED TO CHANGE BEFORE RUNNING IN GCP
            tf_list = index.read_a_posting_list('.', token, bucket_name)
            # print(f'tf_list: {tf_list}')
            if tf_list is not None:
                for doc_id, tf in tf_list:
                    norm_tf = tf / len(index.id_title_dict[doc_id])
                    if doc_id in total_ranking_score.keys():
                        total_ranking_score[doc_id] += weight * norm_tf
                    else:
                        if pv_dict.get(doc_id) is None:
                            pv_dict[doc_id] = 0
                        # if id == 44074302:
                        #   print(f'pv: {pv_dict[id]}')
                        #   print(f'pr: {pr_dict[id]}')
                        total_ranking_score[doc_id] = (
                                    (weight * norm_tf) + (pr_dict[doc_id] * w_pr) + (pv_dict[doc_id] * w_pv))

        most_rel_by_title = sorted([(doc_id, score) for doc_id, score in total_ranking_score.items()],
                                   key=lambda x: x[1], reverse=True)
        # print('finish title')
        return most_rel_by_title