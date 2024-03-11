from collections import Counter
import math
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Define stopwords

bucket_name = '318305570_207757535_bucket'
PROJECT_ID = 'irfinalproject-415108'
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "describe", "consider"]

all_stopwords = english_stopwords.union(corpus_stopwords)

class sim_calc:
    """ A class containing methods for calculating document similarity using different algorithms."""
    
    @staticmethod
    def tokenize(text):
        """
           Tokenize the given text.

           Args:
           - text (str): The text to tokenize.

           Returns:
           - list: A list of tuples containing tokens and their weights.
        """
        if '"' in text:
            RE_PHRASE = re.compile(r'\\*"(.*?)\\*"')
            phrases = [match.group(1) for match in RE_PHRASE.finditer(text)]
            # Remove certain words from the phrases
            filtered_phrases = [
                [' '.join(word for word in phrase.split() if word.lower() not in all_stopwords) for phrase in phrases][
                    0].lower()]
            RE_WORD = re.compile(r'\b\w+\b')
            # Exclude phrases and stopwords from the words
            tokens = [(word, 3) if word in filtered_phrases[0] else (word, 0) for word in
                      RE_WORD.findall(text.lower()) if word not in all_stopwords]
            return tokens
        else:
            RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
            tokens = [(token.group(), 1.0) for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
            return tokens

    @staticmethod
    def stemming(tokens):
        """
            Perform stemming on the given tokens.

            Args:
            - tokens (list): A list of tuples containing tokens and their weights.

            Returns:
            - list: A list of tuples containing stemmed tokens and their weights.
        """
        stemmer = PorterStemmer()
        stemmed_tokens = [(stemmer.stem(x[0]), x[1]) for x in tokens]
        return stemmed_tokens

    @staticmethod
    def get_synonyms(word):
        """
            Get synonyms for the given word.

            Args:
            - word (str): The word to find synonyms for.

            Returns:
            - list: A list of tuples containing synonyms and their weights.
        """
        stemmer = PorterStemmer()
        synonyms = set()
        stemmed_word = stemmer.stem(word)
        for syn in wordnet.synsets(word):
            for i in range(0, 2):
                # Get the normalized form of the word from the lemma
                if len(syn.lemmas()) > i:
                    normalized_word = syn.lemmas()[i].name()
                    # Check if the word contains any signs
                    if normalized_word.isalpha():
                        normalized_word = normalized_word.lower()
                        if normalized_word != word and stemmer.stem(normalized_word) != stemmed_word:
                            synonyms.add((normalized_word, 0.5))
                            return list(synonyms)
                else:
                    break
        return list(synonyms)

    @staticmethod
    def find_hyponyms_and_hypernyms(word):
        hyponyms = []
        synsets = wordnet.synsets(word)
        for synset in synsets:
            try:
                first_hyponym = synset.hyponyms()[0].name()
                first_hypernym_name = first_hyponym.lower()
                if first_hypernym_name != word:
                    hyponyms.append(first_hypernym_name)
            except IndexError:
                pass  # No hyponyms found for the synset
        return hyponyms

    @staticmethod
    def BM25(tokens, index, weight, K=1.2, B=0.75):
        """
            Calculate BM25 score for the given tokens.

            Args:
            - tokens (list): A list of tuples containing tokens and their weights.
            - index: Index object.
            - weight: Weight parameter.
            - K (float): BM25 parameter.
            - B (float): BM25 parameter.

            Returns:
            - list: A list of tuples containing document IDs and their BM25 scores.
        """
        query_counter = Counter([x[0] for x in tokens])
        doc_BM25_value = Counter()
        for token, token_weight in tokens:
            # calc idf for specific token
            if index.df.get(token) is not None:
                token_df = index.df[token]
            else:
                continue
            token_idf = math.log((index.doc_count + 1) / token_df, 10)
            norm_query_token = query_counter[token]
            tf_list = index.read_a_posting_list('.', token, bucket_name)
            for doc_id, tf in tf_list:
                numerator = tf * (K + 1)
                denominator = tf + K * (1 - B + ((B * index.doc_tf_idf_doc_len[doc_id][1]) / index.doc_len_avg))
                query_factor = ((K + 1) * norm_query_token) / (K + norm_query_token)
                if doc_id in doc_BM25_value:
                    doc_BM25_value[doc_id] += (token_weight * weight * (token_idf * (numerator / denominator) * query_factor))
                else:
                    doc_BM25_value[doc_id] = (token_weight * weight * (token_idf * (numerator / denominator) * query_factor))
        return doc_BM25_value.most_common()

    @staticmethod
    def cosin(tokens, index, w_cosine):
        """
            Calculate cosine similarity for the given tokens.

            Args:
            - tokens (list): A list of tuples containing tokens and their weights.
            - index: Index object.
            - w_cosine: Weight parameter.

            Returns:
            - list: A list of tuples containing document IDs and their cosine similarity scores.
        """
        query_counter = Counter([x[0] for x in tokens])
        numerator_value_per_doc = Counter()
        denominator_query = 0
        for token, token_weight in tokens:
            if index.df.get(token) is not None:
                token_df = index.df[token]
            else:
                continue
            token_idf = math.log(index.doc_count / token_df, 10)
            norm_query_token = query_counter[token] / len(tokens)
            denominator_query += math.pow(norm_query_token * token_idf, 2)
            tf_list = index.read_a_posting_list('.', token, bucket_name)
            for doc_id, tf in tf_list:
                norm_tf = tf / index.doc_tf_idf_doc_len[doc_id][1]
                numerator_value_per_doc[doc_id] += (token_weight * norm_tf * token_idf) * (norm_query_token * token_idf)
        cosin_doc_val = Counter()
        for doc_id in numerator_value_per_doc.keys():
            denominator_doc = index.doc_tf_idf_doc_len[doc_id][0] / index.doc_tf_idf_doc_len[doc_id][1]
            cosin_doc_val[doc_id] = (w_cosine * (numerator_value_per_doc[doc_id] / (math.sqrt(denominator_query) * denominator_doc)))
        return cosin_doc_val.most_common()

    @staticmethod
    def compound_ranking(alpha, beta, tokens, index, pr_dict, pv_dict, w_pr):
        """
            Perform compound ranking using BM25 and cosine similarity.

            Args:
            - alpha (float): Weight for cosine similarity.
            - beta (float): Weight for BM25.
            - tokens (list): A list of tuples containing tokens and their weights.
            - index: Index object.
            - pr_dict: Placeholder dictionary.
            - pv_dict: Placeholder dictionary.
            - w_pr: Placeholder parameter.

            Returns:
            - list: A list of tuples containing document IDs and their compound ranking scores.
        """
        bm25_result = sim_calc.BM25(tokens, index, beta, pr_dict, pv_dict, w_pr)
        cosin_result = sim_calc.cosin(tokens, index, alpha)
        compound_counter = Counter()
        for tup in cosin_result:
            compound_counter[id] = (alpha * bm25_result[tup[0]]) + (beta * cosin_result[tup[0]])
        return compound_counter.most_common()

    @staticmethod
    def anchor(index, tokens, weight):
        """
            Calculate anchor scores for the given tokens.

            Args:
            - index: Index object.
            - tokens (list): A list of tuples containing tokens and their weights.
            - weight: Weight parameter.

            Returns:
            - list: A list of tuples containing document IDs and their anchor scores.
        """
        anchor_counter = Counter()
        for token, token_weight in tokens:
            try:
                if index.df[token] is not None or index.df[token] != 0:
                    tf_list = index.read_a_posting_list('.', token, bucket_name)
                    for doc_id, tf in tf_list:
                            if doc_id in anchor_counter:
                                anchor_counter[doc_id] += token_weight * weight * tf
                            else:
                                anchor_counter[doc_id] = token_weight * weight * tf
                else:
                    continue
            except:
                pass
        return anchor_counter.most_common()

    @staticmethod
    def get_title_binary_score(index, tokens, weight):
        """
            Calculate binary scores based on title matching for the given tokens.

            Args:
            - index: Index object.
            - tokens (list): A list of tuples containing tokens and their weights.
            - weight: Weight parameter.

            Returns:
            - list: A list of tuples containing document IDs and their binary scores.
        """
        title_counter = Counter()
        for token, token_weight in tokens:
            if index.df.get(token) is not None:
                tf_list = index.read_a_posting_list('.', token, bucket_name)
                if tf_list is not None:
                    for doc_id, tf in tf_list:
                        norm_tf = tf / index.id_title_dict[doc_id][1]
                        if doc_id in title_counter:
                            title_counter[doc_id] += token_weight * weight * norm_tf
                        else:
                            title_counter[doc_id] = token_weight * weight * norm_tf
            else:
                continue
        return title_counter.most_common()
