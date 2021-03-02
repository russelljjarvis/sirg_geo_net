import shelve
from auxillary_methods import author_to_coauthor_network, network
from grab_js import coords_of_target_university, university_data_frame
import networkx as nx
import pickle
import requests
import matplotlib.pyplot as plt
#from scholarly import scholarly


import requests
import crossref_commons.retrieval

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


author_dict = {
	"Brian H Smith":"Arizona State University, Tempe, AZ, USA",
	"Christian Rabeling":"Arizona State University, Tempe, AZ, USA",
	"Jon F Harrison":"Arizona State University, Tempe, AZ, USA",
	"Juergen Liebig":"Arizona State University, Tempe, AZ, USA",
	"Stephen C Pratt":"Arizona State University, Tempe, AZ, USA",
	"Jennifer H Fewell":"Arizona State University, Tempe, AZ, USA",
	"Arianne J Cease":"Arizona State University, Tempe, AZ, USA",
	"Gro V Amdam":"Arizona State University, Tempe, AZ, USA",
}
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
def orcid_to_address(orcid_id):
	response = requests.get('https://pub.orcid.org/v3.0/'+str(orcid_id)+str('/employments'), headers=headers)#, params=params)
	try:
		temp = response.json()
		address = temp['affiliation-group'][0]['summaries'][0]['employment-summary']['organization']
		print(address, 'o address')
		return address
	except:
		return None
#if 'existing_nodes' not in locals():
#	from grab_js import *
#	existing_nodes = [i['id'] for i in js_data_contents['authorGraph.js']['nodes'] ]
def author_to_gaffiliations(NAME):
	from scholarly import scholarly, ProxyGenerator

	pg = ProxyGenerator()
	pg.Tor_External(tor_sock_port=9050, tor_control_port=9051, tor_password="0r4ng3s")
	scholarly.use_proxy(pg)
	search_query = list(scholarly.search_author(NAME))
	if len(search_query):
		sq = search_query[0]
		#citedby = sq['citedby']
		afil = sq['affiliation']
		#print(afil,"g afil")
		return afil
	else:
		return None

def author_to_gh_index(NAME):
	from scholarly import scholarly, ProxyGenerator

	pg = ProxyGenerator()
	pg.Tor_External(tor_sock_port=9050, tor_control_port=9051, tor_password="0r4ng3s")
	scholarly.use_proxy(pg)

	search_query = list(scholarly.search_author(NAME))
	if len(search_query):
		sq = search_query[0]

		#import pdb
		#pdb.set_trace()
		#afil = sq['affiliation']
		#print(afil,"g afil")
		if hasattr(sq,'citedby'):
			return sq['citedby']
		else:
			return None
	else:
		return None
		#res_author_search = scholarly.fill(sq)
		#try:
			#print(res_author_search)
		#afil = res_author_search['affiliation']
			#if len(afil['affiliation']):
			#	print(afil['affiliation'],res_author_search['affiliation'])
		#gaffiliations[key] = afil
		#print(gaffiliations, "\n\n\n")
			#res_author_search['affiliation']

		#with open('gaffilations.p','wb') as f:
		#	pickle.dump(gaffiliations,f)

	#response = requests.get("https://dissem.in/api/search/?authors=" + str(NAME))
	#author_papers = response.json()
	#visit_urls = []
	#coauthors = []
	#titles = []
	#gaffiliations = {}
	#orcids = {}
	#addresses = {}
	'''
	for p in author_papers["papers"]:
		coauthors_ = p["authors"]
		records = p["records"][0]
		if "doi" in records.keys():
			visit_urls.append(records["doi"])
			doi_to_author_affil_list = crossref_commons.retrieval.get_publication_as_json(records["doi"])
			for al in doi_to_author_affil_list["author"]:
				key = al['given']+str(" ")+al['family']
				#if NAME in key:
				if len(al['affiliation']):
					gaffiliations[key] = al['affiliation'][0]['name']
				if "ORCID" in al.keys():
					orcids[key] = al["ORCID"]
					addresses = orcid_to_address(orcids[key])

	'''

					#except:
					#	import pdb
					#	pdb.set_trace()


	#author_to_gaffiliations
	#return gaffiliations,orcids,addresses
#print([name for name in mg.nodes])




#print(len(mg.nodes))
# https://curl.trillworks.com/#python

"""
plus_initial=NAME['name']['first']
initial = plus_initial.split(" ")
if len(initial)==2:
first_name = initial[0]+str("+")+initial[1][0]
else:
first_name = plus_initial
name = first_name+str("+")+NAME['name']['last']
"""

def name_parse_dict(co):
	plus_initial=co['name']['first']
	initial = plus_initial.split(" ")
	if len(initial)==2:
		first_name = initial[0]+str("+")+initial[1][0]
	else:
		first_name = plus_initial
	name = first_name+str("+")+str('AND')+str("+")+co['name']['last']
	return name
def name_parse_string(co):
	#plus_initial=co['name']['first']
	first = co.split(" ")
	if len(first)==3:
		first_name = first[0]#+str("+")+initial[1][0]
		initial = first[1]#.split(" ")
		last = first[2]#.split(" ")
		name = first_name+str("+")+initial+str("+")+last
	else:
		first_name = first[0]#+str("+")+initial[1][0]
		last = first[1]#.split(" ")
		name = first_name+str("+")+str('AND')+str("+")+last
	return name

		#if len(last_and_initial):
		#	initial = last_and_initial[0]
		#	print(last_and_initial)
		#	last = last_and_initial[1]#+str("+")+initial[1][0]

		#else:
			#last = first[1]



headers = {
	'Accept': 'application/vnd.orcid+json',
}


def name_to_orcid_to_address(NAME):
	if type(NAME) is type(dict()):
		name = name_parsing(NAME)
	if type(NAME) is type(""):
		name = name_parse_string(NAME)
	params = (
		('q',name),
	)
	response = requests.get('https://pub.orcid.org/v3.0/search/', headers=headers, params=params)
	temp = response.json()
	if len(temp):
		orcid_id = temp['result'][0]['orcid-identifier']['path']
		response = requests.get('https://pub.orcid.org/v3.0/'+str(orcid_id)+str('/employments'), headers=headers)#, params=params)
		temp = response.json()
		try:
			org_name =  temp['affiliation-group'][0]['summaries'][0]['employment-summary']['organization']['name']
			org_address = temp['affiliation-group'][0]['summaries'][0]['employment-summary']['organization']['address']
			full_address = org_name+str(" ")+org_address['city']+str(" ")+org_address['region']+str(" ")+org_address['country']
		except:
			full_address = None
		return full_address
	else:
		return None

def author_to_oaffiliations(NAME):
	#response = requests.get("https://dissem.in/api/search/?authors=" + str(NAME))
	#author_papers = response.json()
	#o_addresses = {}
	#for p in author_papers["papers"]:
	#coauthors_ = p["authors"]
	#for co in coauthors_:
	#print(co)
	#name = name_parsing(NAME)
	#print(NAME)
	o_address = name_to_orcid_to_address(NAME)
	print(o_address,'o address')
	#g_affiliations[name] = author_to_gaffiliations(co,g_affiliations)
	#print(g_affiliations[name])

	return o_address
try:
	with open('gaffilations.p','rb') as f:
		gaffiliations = pickle.load(f)
	for k,v in affiliations.items():
		if type(v) is type(list()):
			gaffiliations[k] = v[0]['name']
except:
	pass


try:
	with open('oaffilations.p','rb') as f:
		orcid_affiliations = pickle.load(f)
except:
	pass
try:
	with open('gaffilations.p','rb') as f:
		g_affiliations = pickle.load(f)
except:
	pass
'''
for k,v in orcid_affiliations.items():
	if k in g_affiliations.keys():
		print(k,v,g_affiliations[k])
	else:
		print(k,v)

for k,v in g_affiliations.items():
	if k in orcid_affiliations.keys():
		print(k,v,orcid_affiliations[k])
	else:
		print(k,v)
'''
#import pdb
#pdb.set_trace()

with open('oaffilations.p','rb') as f:
	orcid_affiliations = pickle.load(f)
with open('gaffilations.p','rb') as f:
	g_affiliations = pickle.load(f)

todo = len(mg.nodes)
for name in mg.nodes:

	if name not in orcid_affiliations.keys():
		address = author_to_oaffiliations(name)
		orcid_affiliations[name] = address
		with open('oaffilations.p','wb') as f:
			pickle.dump(orcid_affiliations,f)
		todo-=1

	else:
		todo-=1
		print(name,'done')
	'''
	if name not in g_affiliations.keys():
		address = author_to_gaffiliations(name)
		orcid_affiliations[name] = address
		with open('gaffilations.p','wb') as f:
			pickle.dump(g_affilations,f)
		todo-=1
	'''
	print(todo)

def get_coords():
	orcid_affiliations = { k:v for k,v in orcid_affiliations.items() if type(v) is not type(dict()) }
	orcid_affiliations.update(author_dict)

	with open('ocoords.p','rb') as f:
		orcid_locations = pickle.load(f)

	try:
		with open('gcoords.p','rb') as f:
			g_locations = pickle.load(f)
	except:
		g_locations = {}

	for k,v in orcid_affiliations.items():
		if k not in orcid_locations.keys():
			if v is not None:
				print(v,'in orcid locations')
				orcid_locations[k] = coords_of_target_university(v)
				with open('ocoords.p','wb') as f:
					pickle.dump(orcid_locations,f)

	for k,v in g_affiliations.items():

		if k not in g_locations.keys():
			if v is not None:
				print(v,'in google locations')
				g_locations[k] = coords_of_target_university(v)

			with open('gcoords.p','wb') as f:
				pickle.dump(g_locations,f)

	both_sets_locations = orcid_locations
	orcid_locations.update(g_locations)
	with open('both_sets_locations.p','wb') as f:
		pickle.dump(both_sets_locations,f)

'''
with open('oaffilations.p','rb') as f:
	orcid_affiliations = pickle.load(f)

#with open('gaffilations.p','rb') as f:#
#	g_affiliations = pickle.load(f)

try:
	with open('gaffilations.p','rb') as f:
		gaffiliations = pickle.load(f)
	for k,v in gaffiliations.items():
		if type(v) is type(list()):
			gaffiliations[k] = v[0]['name']
	with open('gaffilations.p','wb') as f:
		pickle.dump(gaffiliations,f)
except:
'''



	#temp['name']+str(" ")+temp['address']
#import pdb
#pdb.set_trace()
#orcid_id = temp['result'][0]['orcid-identifier']['path']

#curl --location --request GET 'https://pub.orcid.org/v3.0/0000-0001-9813-5701/employments' \
#--header 'Accept: application/vnd.orcid+json'
#print(temp)
#import pdb
#pdb.set_trace()
'''
df,b = university_data_frame()
for ind,(count,univ) in enumerate(zip(df["country"], df["university"])):
	try:
		returned = coords_of_target_university(univ[ind])
	except:
		pass
		#import pdb
		#pdb.set_trace()
		#returned = coords_of_target_university(count[ind])

	if returned is not None:
		university_locations[univ[ind]] = returned
	if univ[ind] in university_locations.keys():
		print(university_locations[univ[ind]])
'''
#curl
	#print(university_locations)
#print(university_locations)
#print(exhaustive_author_list)
#yes = set(exhaustive_author_list).intersection(existing_nodes)
#print(yes)
