from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Zoogle Search"
net_id = "Thomas Lee (tcl59), Joseph Choi (jc2493), Ray Wei (ryw23), Joyce Huang (sh2429), Clara Song (cs2274)"

combined_inv_idx = load_idx()
combined_idf_dict = load_idf()
combined_doc_norms = load_norms()
combined_breeds_info = load_breed()

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ""
	else:
		output_message = "Your search: " + query
		results = index_search(query, combined_inv_idx, combined_idf_dict, combined_doc_norms)
		data = results
		breeds = list(combined_breeds_info.keys())
		for i in range(len(results)):
			score = results[i][0]
			msg_id = results[i][1]
			data[i] = str(i+1) + ": " + str(breeds[msg_id]) + "(" + str(score) + ")"
		if (len(data) == 0):
			data = ["no items matched:( try a different keyword!"]
		else:
			data = data[:10]
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)