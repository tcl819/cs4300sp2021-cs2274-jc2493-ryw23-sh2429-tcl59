# Methods to compose HTTP response JSON 
from flask import jsonify
import re
import base64
import json
import numpy as np
import math
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer

treebank_tokenizer = TreebankWordTokenizer()

def http_json(result, bool):
	result.update({ "success": bool })
	return jsonify(result)


def http_resource(result, name, bool=True):
	resp = { "data": { name : result }}
	return http_json(resp, bool)


def http_errors(result): 
	errors = { "data" : { "errors" : result.errors["_schema"] }}
	return http_json(errors, False)

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict 
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)
        
def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def load_idx():
    inv_idx_file_names = ['json_idx.json',
                      'json_idx_bird.json',
                      'json_idx_dog.json',
                      'json_idx_cat.json',
                      'json_idx_fish.json',
                      'json_idx_horse.json',
                      'json_idx_rabbit.json',
                      'json_idx_turtle.json']
    imported_idx = []
    for i in range(0,8):
        with open('data/jsons/inv_idx/' + inv_idx_file_names[i]) as json_file:
            imported_idx.append(json.load(json_file))
    return imported_idx

def load_idf():
    idf_dict_file_names = ['json_idf.json',
                       'json_idf_bird.json', 
                       'json_idf_dog.json',
                       'json_idf_cat.json',
                       'json_idf_fish.json',
                       'json_idf_horse.json',
                       'json_idf_rabbit.json',
                       'json_idf_turtle.json']
    imported_idf = []
    for i in range(0,8):
        with open('data/jsons/idf_dict/' + idf_dict_file_names[i]) as json_file:
            imported_idf.append(json.load(json_file))
    return imported_idf

def load_norms():
    doc_norms_file_names = ['json_norms.json',
                       'json_norms_bird.json', 
                       'json_norms_dog.json',
                       'json_norms_cat.json',
                       'json_norms_fish.json',
                       'json_norms_horse.json',
                       'json_norms_rabbit.json',
                       'json_norms_turtle.json']
    imported_norms = []
    for i in range(0,8):
        with open('data/jsons/doc_norms/' + doc_norms_file_names[i]) as json_file:
            imported_norms.append(json.load(json_file))
    return imported_norms

def load_breed():
    breed_info_file_names = ['json_breed.json',
                       'json_breed_bird.json', 
                       'json_breed_dog.json',
                       'json_breed_cat.json',
                       'json_breed_fish.json',
                       'json_breed_horse.json',
                       'json_breed_rabbit.json',
                       'json_breed_turtle.json']
    imported_breed = []
    for i in range(0,8):
        with open('data/jsons/breed_info/' + breed_info_file_names[i]) as json_file:
            imported_breed.append(json.load(json_file))
    return imported_breed

def load_query_mat():
    with open('data/jsons/json_query_mat.json') as json_file:
        query_mat = json.load(json_file)
    return query_mat

def save_query_mat(query_mat):
    with open('data/jsons/json_query_mat.json', 'w') as json_file:
        json.dump(query_mat, json_file)


def index_search(query, input_doc_mat, index, idf, doc_norms, tokenizer=treebank_tokenizer):
    """
    Search the collection of documents for the given query
    """
    # load query_mat
    query_mat = load_query_mat()

    if query not in query_mat:
        query_vec = query_to_vec(query, index, idf)

        query_mat[query] = dict()
        query_mat[query]['vec'] = query_vec.tolist()
        query_mat[query]['relevant'] = []
        query_mat[query]['irrelevant'] = []
    else:
        query_vec = np.array(query_mat[query]['vec'])
    
    query_norm = np.linalg.norm(query_vec)
    results = []

    for i in range(len(input_doc_mat)):
        doc_vec = input_doc_mat[i]
        norm = np.linalg.norm(input_doc_mat[i])
        score = np.dot(query_vec, input_doc_mat[i])/(norm*query_norm)
        results.append((score, i))
        
    # sort tuple list in descending order by score, then doc_id
    results = sorted(results, key=lambda e: (e[0], -e[1]), reverse=True)

    # save query_mat
    save_query_mat(query_mat)
    
    return results

def first_n_sentences(breed, input_info, n_sentences):
    """
    Parameters: 
        breed: string specifying the name of the breed
        input_info: Dictionary of breeds of a specific animal type
    Returns: string containing the first three sentences of the breed information
    """
    text = input_info[breed]['text']
    regex = '(?<![A-Z])[.!?]\s+(?=[A-Z])'
    match = re.split(regex, text)
    result = ""
    for sentence in range(n_sentences):
        result = result + match[sentence] + ".  "
    return result

def query_to_vec(query, inverted_idx, idf):
    """
    Turns the query into a numpy vector of TF-IDF weights, of length = # of terms

    Parameters:
    query: string, query search
    inverted_idx: dict, inverted index, term frequency in doc for each word
    idf: dict, IDF scores for each word

    Returns: np vector, length = # of terms
    """
    porter_stemmer = PorterStemmer()
    tokenizer = treebank_tokenizer

    index_to_word = list(inverted_idx.keys())
    word_to_index = {t: i for i, t in enumerate(index_to_word)}
    query_vec = np.zeros(len(index_to_word))

    query = query.lower()
    regex = '[a-z]+'
    match = re.findall(regex,query)
    tokens = [porter_stemmer.stem(word) for word in match]

    for tok in tokens:
        if tok in idf:
            ind = word_to_index[tok]
            query_vec[ind] = (1/len(tokens))/idf[tok]
    return query_vec


def create_tf_idf_mat(n_breeds, index, idf):
    """
    Creates a TF-IDF matrix

    Parameters:
    n_breeds: int, number of breeds/docs
    index: dict, inverted index, term frequency in doc for each word
    idf: dict, IDF scores for each word

    Returns:
    # of docs x # of terms TF-IDF numpy matrix
    """
    index_to_word = list(index.keys())
    word_to_index = {t: i for i, t in enumerate(index_to_word)}

    arr = np.zeros((n_breeds, len(index)))

    for word in index:
        if word in idf:
            for tup in index[word]:
                doc_id = tup[0]
                tf = tup[1]
                arr[tup[0]][word_to_index[word]] = tf * idf[word]
    return arr


def rocchio(query, query_mat, input_doc_mat, a=1, b=0.8, c=0.1):
    """
    Precondition: query is in query_mat

    query_mat: dictionary, query -> dictionary
        query_mat[key]['relevant']: list of IDs of relevant docs
        query_mat[key]['irrelevant']: list of IDs of irrelevant docs
        query_mat[key]['vec']: list form of numpy TF-IDF vector of query (NOT stored as numpy array)

    Parameters:
    query: string
    input_doc_mat: # of docs x # of terms numpy matrix of TF-IDF weights

    Returns:
    numpy TF-IDF vector
    """
    # load query_mat
    relevant = query_mat[query]['relevant']
    irrelevant = query_mat[query]['irrelevant']
    vec = np.array(query_mat[query]['vec'])

    relevant_w = 0 if len(relevant) == 0 else 1/len(relevant)
    irrelevant_w = 0 if len(irrelevant) == 0 else 1/len(irrelevant)

    rel_vec = np.zeros(vec.shape)
    irrel_vec = np.zeros(vec.shape)

    for doc_id in relevant:
        rel_vec += input_doc_mat[doc_id]
    for doc_id in irrelevant:
        irrel_vec += input_doc_mat[doc_id]
    
    result = a * vec + b * relevant_w * rel_vec - c * irrelevant_w * irrel_vec
    return result.clip(min=0)


def process_results(query, index, idf, doc_norms, breed_info, tokenizer=treebank_tokenizer):
    """
    Returns: list of dictionaries as shown below for the top ten results...
             'name': name of the breed,
             'text': The first three sentences from the text from petguides.com,
             'score': Similarity score
             'URL_petguide': URL leading to the petguides website,
             'URL_image': URL for the image from petguides.com
    """

    # list of results from searching
    final_results = []
    # list of breed names
    breeds = list(breed_info.keys())
    # # of docs x # of terms tf-idf numpy matrix
    input_doc_mat = create_tf_idf_mat(len(breeds), index, idf)

    # results from index search
    raw_results = index_search(query, input_doc_mat, index, idf, doc_norms, tokenizer)

    # return top 10 results with fields for name, text to output, URL to petguides, URL for image  
    for result in raw_results:
        result_dict = {}
        result_dict['raw_name'] = breeds[result[1]]
        result_dict['name'] = breed_info[breeds[result[1]]]['name']
        result_dict['text'] = first_n_sentences(breeds[result[1]], breed_info, 4)
        result_dict['score'] = result[0]
        result_dict['URL_petguide'] = "#" if not 'page_url' in list(breed_info[breeds[result[1]]].keys()) else breed_info[breeds[result[1]]]['page_url']
        result_dict['URL_image'] = "#" if not 'image_url' in list(breed_info[breeds[result[1]]].keys()) else breed_info[breeds[result[1]]]['image_url']
        result_dict['type'] = breed_info[breeds[result[1]]]['type']
        final_results.append(result_dict)
    
    return final_results