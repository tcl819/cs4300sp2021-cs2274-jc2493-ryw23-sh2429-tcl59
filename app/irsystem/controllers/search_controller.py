from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Zoogle Search"
net_id = "Thomas Lee (tcl59), Joseph Choi (jc2493), Ray Wei (ryw23), Joyce Huang (sh2429), Clara Song (cs2274)"

inv_idx = load_idx()
idf_dict = load_idf()
norms = load_norms()
breeds_info = load_breed()

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	
	# DO NOT CHANGE FROM 0 RIGHT NOW
	filter_id = 0 # 0 for all, 1-7 for the seven breeds
	index = inv_idx[filter_id]
	idf = idf_dict[filter_id]
	norm = norms[filter_id]
	info = breeds_info[filter_id]

	is_bird_checked = 'bird' in request.args
	is_cat_checked = 'cat' in request.args
	is_dog_checked = 'dog' in request.args
	is_fish_checked = 'fish' in request.args
	is_horse_checked = 'horse' in request.args
	is_rabbit_checked = 'rabbit' in request.args
	is_turtle_checked = 'turtle' in request.args
	none_checked = not(is_bird_checked) and not(is_cat_checked) and not(is_dog_checked) and not(is_fish_checked) and not(is_horse_checked) and not(is_rabbit_checked) and not(is_turtle_checked)

	if not query:
		data = []
		output_message = ""
	else:
		output_message = "Your search: " + query
		# results = index_search(query, combined_inv_idx, combined_idf_dict, combined_doc_norms)
		results = process_results(query, index, idf, norm, info)
		data = []
		count = 0
		for i in range(len(results)):
			if (none_checked or results[i]['type'] in request.args):
				count = count + 1
				dic = dict()
				dic['raw_name'] = results[i]['raw_name']
				dic['name'] = results[i]['name']
				dic['score'] = results[i]['score']
				dic['text'] = results[i]['text']
				dic['pos'] = count
				dic['page_url'] = results[i]['URL_petguide']
				dic['image_url'] = results[i]['URL_image']
				data.append(dic)
		if (len(data) == 0):
			data = []
		else:
			data = data[:10]

	feedback = []
	for i in range(10):
		f = request.args.get('rating-'+str(i+1))
		if f is not None:
			feedback.append(f)
	
	if feedback:
		# load query_mat
		query_mat = load_query_mat()
		docs = list(info.keys())
		breed_to_index = {t: i for i, t in enumerate(docs)}
		for i in range(10):
			# breed_name = data[i]['name']
			doc_id = breed_to_index[data[i]['raw_name']]
			
			if feedback[i] == "1":
				if doc_id not in query_mat[query]['relevant']:
					query_mat[query]['relevant'].append(doc_id)
				if doc_id in query_mat[query]['irrelevant']:
					query_mat[query]['irrelevant'].remove(doc_id)
			elif feedback[i] == "-1":
				if doc_id not in query_mat[query]['irrelevant']:
					query_mat[query]['irrelevant'].append(doc_id)
				if doc_id in query_mat[query]['relevant']:
					query_mat[query]['relevant'].remove(doc_id)
		
		input_doc_mat = create_tf_idf_mat(len(info), index, idf)
		updated_vec = rocchio(query, query_mat, input_doc_mat)
		query_mat[query]['vec'] = updated_vec.tolist()
		save_query_mat(query_mat)


	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, query=query, feedback=feedback)