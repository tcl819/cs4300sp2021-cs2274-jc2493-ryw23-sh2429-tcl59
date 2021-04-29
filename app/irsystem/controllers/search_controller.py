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

	if not query:
		data = []
		output_message = ""
	else:
		output_message = "Your search: " + query
		# results = index_search(query, combined_inv_idx, combined_idf_dict, combined_doc_norms)
		results = process_results(query, index, idf, norm, info)
		data = []
		for i in range(len(results)):
			data.append(dict())
			data[i]['name'] = results[i]['name']
			data[i]['score'] = results[i]['score']
			data[i]['text'] = results[i]['text']
			data[i]['pos'] = i+1
			data[i]['page_url'] = results[i]['URL_petguide']
			data[i]['image_url'] = results[i]['URL_image']
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
		print(feedback)
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, query=query, feedback=feedback)