# Methods to compose HTTP response JSON 
from flask import jsonify
import re
import base64
import json
import numpy as np
import math
from nltk.tokenize import TreebankWordTokenizer

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
    with open('data/jsons/json_idx.json') as json_file:
        imported_idx = json.load(json_file)
    return imported_idx

def load_idf():
    with open('data/jsons/json_idf.json') as json_file:
        imported_idf = json.load(json_file)
    return imported_idf

def load_norms():
    with open('data/jsons/json_norms.json') as json_file:
        imported_norms = json.load(json_file)
    return imported_norms

def load_breed():
    with open('data/jsons/json_breed.json') as json_file:
        imported_breed = json.load(json_file)
    return imported_breed

treebank_tokenizer = TreebankWordTokenizer()

def index_search(query, index, idf, doc_norms, tokenizer=treebank_tokenizer):
    """Search the collection of documents for the given query"""
    # Solve for query term frequencies
    query_freq = dict()
    for tok in tokenizer.tokenize(query.lower()):
        if tok in query_freq:
            query_freq[tok] += 1
        else:
            query_freq[tok] = 1
    
    # Solve for query norm
    q_norm = 0
    for word in query_freq:
        if word in idf:
            idf_val = idf[word]
            q_norm += (idf_val*query_freq[word])**2
    q_norm = math.sqrt(q_norm)
    
    # calculate numerator values of cosine similarity for each document
    numerators = dict() # doc_id -> cumulative numerator
    for word in index:
        if word in idf:
            for tup in index[word]:
                doc_id = tup[0]
                if word in query_freq:
                    if doc_id not in numerators:
                        numerators[doc_id] = 0
                    numerators[doc_id] += query_freq[word] * tup[1] * idf[word]**2
    
    results = []
    # divide each numerator by the appropriate denominator, add to list in tuple form
    for doc_id in numerators:
        denom = q_norm * doc_norms[doc_id]
        score = numerators[doc_id] / denom
        results.append((score, doc_id))
    # sort tuple list in descending order by score, then doc_id
    results = sorted(results, key=lambda e: (e[0], -e[1]), reverse=True)
    return results

def first_n_sentences(breed, input_info, n_sentences):
    """
    Parameters: 
        breed: string specifying the name of the breed
        input_info: Dictionary of breeds of a specific animal type
    Returns: string containing the first three sentences of the breed information """
    text = input_info[breed]['text']
    regex = '(?<![A-Z])[.!?]\s+(?=[A-Z])'
    match = re.split(regex, text)
    result = ""
    for sentence in range(n_sentences):
        result = result + match[sentence] + ".  "
    return result

def process_results(index_search_function, query, index, idf, doc_norms, breed_info, tokenizer=treebank_tokenizer):
    """Returns: list of dictionaries as shown below for the top ten results...
             'name': name of the breed,
             'text': The first three sentences from the text from petguides.com,
             'URL_petguide': URL leading to the petguides website,
             'URL_image': URL for the image from petguides.com"""

    final_results = []
    breeds = list(breed_info.keys())
    raw_results = index_search_function(query, index, idf, doc_norms, tokenizer)
    # return top 10 results with fields for name, text to output, URL to petguides, URL for image  
    for result in raw_results:
        result_dict = {}
        result_dict['name'] = breed_info[breeds[result[1]]]['name']
        result_dict['text'] = first_n_sentences(breeds[result[1]], breed_info, 4)
        result_dict['score'] = result[0]
        result_dict['URL_petguide'] = "#" if not 'page_url' in list(breed_info[breeds[result[1]]].keys()) else breed_info[breeds[result[1]]]['page_url']
        result_dict['URL_image'] = "#" if not 'image_url' in list(breed_info[breeds[result[1]]].keys()) else breed_info[breeds[result[1]]]['image_url']
        final_results.append(result_dict)
    
    return final_results