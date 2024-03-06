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
from math import *
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords

PROJECT_ID = 'irfinalproject-415108'
def get_bucket(bucket_name):
    return storage.Client(PROJECT_ID).bucket(bucket_name)

def _open(path, mode, bucket=None):
    if bucket is None:
        return open(path, mode)
    return bucket.blob(path).open(mode)

# Let's start with a small block size of 30 bytes just to test things out. 
BLOCK_SIZE = 1999998

class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._file_gen = (_open(str(self._base_dir / f'{name}_{i:03}.bin'), 
                                'wb', self._bucket) 
                          for i in itertools.count())
        self._f = next(self._file_gen)
           
    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:  
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            name = self._f.name if hasattr(self._f, 'name') else self._f._blob.name
            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """
    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            f_name = str(self._base_dir / f_name)
            if f_name not in self._open_files:
                self._open_files[f_name] = _open(f_name, 'rb', self._bucket)
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)
  
    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False 

TUPLE_SIZE = 6       # We're going to pack the doc_id and tf values in this 
                     # many bytes.
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


class InvertedIndex:  
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        self.pre_calc_tf_idf = OrderedDict()
        self.pre_calc_cossin = defaultdict(list)
        self.doc_count = 0
        self.doc_len_avg = 0
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally), 
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of 
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are 
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents 
        # the number of bytes from the beginning of the file where the posting list
        # starts. 
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)



    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage 
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name, bucket_name=None):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name, bucket_name)

    def _write_globals(self, base_dir, name, bucket_name):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'wb', bucket) as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self, base_dir, bucket_name=None):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                    tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def read_a_posting_list(self, base_dir, w, bucket_name=None):
        posting_list = []
        if not w in self.posting_locs:
            return posting_list
        with closing(MultiFileReader(base_dir, bucket_name)) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * TUPLE_SIZE)
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
                tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl
        
        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl: 
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
            bucket = None if bucket_name is None else get_bucket(bucket_name)
            with _open(path, 'wb', bucket) as f:
                pickle.dump(posting_locs, f)
        return bucket_id

    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter(".", bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            writer.upload_to_gcp()
            InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name)
        return bucket_id

    @staticmethod
    def read_index(base_dir, name, bucket_name=None):
        path = str(Path(base_dir) / f'{name}.pkl')
        bucket = None if bucket_name is None else get_bucket(bucket_name)
        with _open(path, 'rb', bucket) as f:
            return pickle.load(f)


class pre_calc:
    # def __init__(self):

    @staticmethod
    def tf_idf(doc_id, index, token):
        N = index.doc_count
        if index.df.get(token) is not None:
            df = index.df[token]
            idf = math.log(N / df, 10)
            tf_list = index._posting_list[token]
            for id, tf in tf_list:
                if id == doc_id:
                    index.pre_calc_tf_idf[(doc_id, token)] = tf * idf
                    break

    @staticmethod
    def tokenize(text, STEMMING=False):
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        all_stopwords = english_stopwords.union(corpus_stopwords)

        tokens = [token.group() for token in RE_WORD.finditer(text.lower())]

        if STEMMING:
            stemmer = PorterStemmer()
            list_of_tokens = [stemmer.stem(x) for x in tokens if x not in all_stopwords]
        else:
            list_of_tokens = [x for x in tokens if x not in all_stopwords]

        return list_of_tokens

    @staticmethod
    def pre_cosin(tokens_doc_pairs, index):
        for tokens, doc_id in tokens_doc_pairs.collect():
            for token in tokens:
                if index.df.get(token) is not None:
                    calc = index.pre_calc_cossin[doc_id][0] + math.pow(index.pre_calc_tf_idf[(doc_id, token)],2)
                    index.pre_calc_cossin[doc_id] = (calc, index.pre_calc_cossin[doc_id][1])


# class sim_calc:
#     @staticmethod
#     def BM25(tokens, index, K=1.2, B=0.75):
#         query_counter = Counter(tokens)
#         BM25_value_per_doc = Counter()
#         for token in tokens:
#             # calc idf for specific token
#             if index.df.get(token) is not None:
#                 token_df = index.df[token]
#             else:
#                 continue
#             token_idf = math.log(index.doc_count / token_df, 10)
#             # loading posting list with (word, (doc_id, tf))
#             tf_list = index._posting_list[token]
#             for id, tf in tf_list:
#                 # normalized tf (by the length of document)
#                 numerator = tf * (K + 1)
#                 denominator = tf + K * (1 - B + (B * index.pre_calc_cosine[id][1]) / index.doc_len_avg)
#                 query_factor = ((K + 1) * query_counter[token]) / (K + query_counter[token])
#                 doc_BM25_value[id] += token_idf * (numerator / denominator) * query_factor
#         sorted_doc_BM25_value = doc_BM25_value.most_common()
#         return sorted_doc_BM25_value
#
    # def cosin(tokens, index):
    #     query_counter = Counter(tokens)
    #     numerator_value_per_doc = Counter()
    #     denominator_query = 0
    #     for token in tokens:
    #         denominator_query += math.pow(query_counter[token], 2)
    #         tf_list = index._posting_list[token]
    #         for id, tf in tf_list:
    #             numerator_value_per_doc[id] += tf * query_counter[token]
    #     cosin_doc_val = Counter()
    #     for id in numerator_value_per_doc.keys():
    #         denominator_doc = index.pre_calc_cossin[doc_id][0]
    #         cosin_doc_val[id] = numerator_value_per_doc[id] / math.sqrt(denominator_doc * denominator_query)
    #     sorted_cosin_doc_val = cosin_doc_val.most_common()
    #     return sorted_cosin_doc_val
#
#     def compound_ranking(alpha, tokens, index):
#         bm25_result = BM25(tokens, index)
#         cosin_result = cosin(tokens, index)
#         compound_counter = Counter()
#         for id in cosin_result.keys():
#             compound_counter[id] = (alpha * bm25_result[id]) + ((1 - alpha) * cosin_result[id])
#         sorted_compound_counter = compound_counter.most_common()
#         return sorted_compound_counter