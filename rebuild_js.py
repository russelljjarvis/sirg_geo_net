import shelve
from auxillary_methods import author_to_coauthor_network, network
import networkx as nx
import pickle
import requests
#from auxillary_methods import data_shade
import matplotlib.pyplot as plt
from scholarly import scholarly

#fig = data_shade(mg)
import crossref_commons.retrieval
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
				with open('data/old_pickle.p','rb') as f:
					old_pickle = pickle.load(f)
				g = old_pickle[k]
				nl = g.nodes
				exhaustive_author_list.extend(list(nl))
				db[k]['g'] = g
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
def author_to_affiliations(NAME):
	try:
		with open('affilations.p','rb') as f:
			affiliations = pickle.load(f)
		for k,v in affiliations.items():
			if type(v) is type(list()):
				affiliations[k] = v[0]['name']
	except:
		pass

	response = requests.get("https://dissem.in/api/search/?authors=" + str(NAME))
	author_papers = response.json()
	visit_urls = []
	coauthors = []
	titles = []
	affiliations = {}
	orcids = {}
	for p in author_papers["papers"]:
		coauthors_ = p["authors"]
		records = p["records"][0]
		if "doi" in records.keys():
			visit_urls.append(records["doi"])
			doi_to_author_affil_list = crossref_commons.retrieval.get_publication_as_json(records["doi"])
			for al in doi_to_author_affil_list["author"]:
				key = al['given']+str(" ")+al['family']
				#if key not in affiliations.keys():
				if len(al['affiliation']):
					affiliations[key] = al['affiliation'][0]['name']
				if "ORCID" in al.keys():
					orcids[key] = al["ORCID"]
				#if not len(al['affiliation']):
				search_query = list(scholarly.search_author(key))
				#sq = search_query[0]
				if len(search_query):
					sq = search_query[0]
					res_author_search = scholarly.fill(sq)
					afil = res_author_search['affiliation']
					#if "university" in afil or "state" in afil or "universidad" in afil or "college" in afil or "school" in afil:
					if len(al['affiliation']):
						#if al['affiliation'] in res_author_search['affiliation']:
						print(al['affiliation'],res_author_search['affiliation'])
					affiliations[key] = res_author_search['affiliation']
							#print(affiliations[key],key)
							#print(affiliations)

	with open('affilations.p','wb') as f:
		pickle.dump(affiliations,f)
	return affiliations
print([name for name in mg.nodes])
#for name in mg.nodes:
#	print(name)
#	affils = author_to_affiliations(name)

with open('affilations.p','rb') as f:
	affiliations = pickle.load(f)
for k,v in affiliations.items():
	if type(v) is type(list()):
		affiliations[k] = v[0]['name']
with open('affilations.p','wb') as f:
	pickle.dump(affiliations,f)

from grab_js import coords_of_target_university, university_data_frame

university_locations = {}
for k,v in affiliations.items():
	#if type(v) is type(list()):
	#affiliations[k]# = v[0]['name']
	print(k,v)

print(len(mg.nodes))
df,b = university_data_frame()
for ind,(count,univ) in enumerate(zip(df["country"], df["university"])):

	#import pdb
	#pdb.set_trace()
	#university_locations[k] = coords_of_target_university(univ[ind])
	print(univ[ind])
	returned = coords_of_target_university(univ[ind])
	if returned is not None:
		university_locations[univ[ind]] = returned
	if univ[ind] in university_locations.keys():
		print(university_locations[univ[ind]])
	#print(university_locations)
#print(university_locations)
#print(exhaustive_author_list)
#yes = set(exhaustive_author_list).intersection(existing_nodes)
#print(yes)
