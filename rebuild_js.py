import shelve
from auxillary_methods import author_to_coauthor_network, network
import networkx as nx
import pickle

#affiliation_id_dict={'affiliation':afil,'id':Author_name}
#js_data_contents['data/js/authorGraph.js'] =

# need to repopulate the file authorGraph js which consists of author affiliation
# pairs.
#name_to_place,name_to_affil = sirg_author_locations()

#for k,v in name_to_affil.items():
#	js_data_contents['authorGraph.js']['nodes'].append({'affiliation':v[0],'id':k})

author_list = [
    "Brian H Smith",
    "Christian Rabeling",
    "Jon F Harrison",
    "Juergen Liebig",
    "Stephen C Pratt",
    "Jennifer H Fewell",
    "Arianne J Cease",
    "Gro V Amdam",
]
def update_data_sources():
	with shelve.open("fast_graphs_splash.p") as db:
		done = set( k for k,v in db.items() )
		al = set(author_list)
		missing = al-done

		for miss in missing:
			db[miss] = {}
			g, df = author_to_coauthor_network(miss)
			db[miss]['g'] = g
			db[miss]['df'] = df
		'''
		done below
		for k,v in db.items():
			if type(v) is type(dict()):
				if 'g' not in v.keys():
					g, df = author_to_coauthor_network(k)
					db[k]['g'] = g
					db[k]['df'] = df
			print('g' in v.keys())
		'''
		db.close()


	to_mega_net = []
	old_pickle = {}
	with shelve.open("fast_graphs_splash.p") as db:
		exhaustive_author_list = []
		for k,v in db.items():
			if 'g' not in db[k].keys():
				g, df = author_to_coauthor_network(k)
				db[k]['g'] = g
				db[k]['df'] = df

			try:
				nl = v['g'].nodes
				exhaustive_author_list.extend(list(nl))
			except:
				with open('old_pickle.p','rb') as f:
					old_pickle = pickle.dump(f)
				g = old_pickle[k]
				#g, df = author_to_coauthor_network(k)
				nl = g.nodes
				exhaustive_author_list.extend(list(nl))
				db[k]['g'] = g
				#old_pickle[k] = g
			to_mega_net.append(g)
		db.close()
		for i,mn in enumerate(to_mega_net):
			if i==0:
				mg = mn
			else:
				mg = nx.compose(mg,mn)

	return mg,old_pickle

try:
	with open('old_pickle.p','rb') as f:
		old_pickle = pickle.load(f)
	with open('mega_net.p','rb') as f:
		mg = pickle.load(f)
except:
	mg,old_pickle = update_data_sources()
	with open('old_pickle.p','wb') as f:
		pickle.dump(old_pickle,f)
	with open('mega_net.p','wb') as f:
		pickle.dump(mg,f)

#if 'existing_nodes' not in locals():
#	from grab_js import *
#	existing_nodes = [i['id'] for i in js_data_contents['authorGraph.js']['nodes'] ]

from auxillary_methods import data_shade
import matplotlib.pyplot as plt
#fig = data_shade(mg)
import crossref_commons.retrieval
def author_to_affiliations(NAME):
    response = requests.get("https://dissem.in/api/search/?authors=" + str(NAME))
    author_papers = response.json()
    visit_urls = []
    coauthors = []
    titles = []
    affilations = {}
    for p in author_papers["papers"]:
        coauthors_ = p["authors"]
        records = p["records"][0]
        if "doi" in records.keys():
            visit_urls.append(records["doi"])
            doi_to_affil = crossref_commons.retrieval.get_publication_as_json(records["doi"])
            key = stored['author'][0]['given']+stored['author'][0]['family']
            affilations[key] = doi_to_affil['author'][0]['affiliation']
    return affilations

#print(exhaustive_author_list)
#yes = set(exhaustive_author_list).intersection(existing_nodes)
#print(yes)
