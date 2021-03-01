import pickle
with open('missing_person_name.p','rb') as f:
	missing_person_name = pickle.load(f)
from rebuild_js import author_to_gaffiliations, author_to_gh_index
from grab_js import coords_of_target_university, university_data_frame

try:
	with open('gad.p','rb') as f:
		gad = pickle.load(f)

except:

	from scholarly import scholarly, ProxyGenerator
	pg = ProxyGenerator()
	pg.Tor_External(tor_sock_port=9050, tor_control_port=9051, tor_password="0r4ng3s")
	scholarly.use_proxy(pg)
	print('proxy enabled')

	ga = []
	gad = {}
	for name in missing_person_name:
		gad[name] = author_to_gaffiliations(name)
		ga.append(author_to_gaffiliations(name))

	with open('ga.p','wb') as f:
		pickle.dump(ga,f)
	with open('gad.p','wb') as f:
		pickle.dump(gad,f)
	with open('mega_net.p','rb') as f:
		mg = pickle.load(f)

	from scholarly import scholarly, ProxyGenerator
	pg = ProxyGenerator()
	pg.Tor_External(tor_sock_port=9050, tor_control_port=9051, tor_password="0r4ng3s")
	scholarly.use_proxy(pg)
	print('proxy enabled')

	for name in mg.nodes:
		gad[name] = author_to_gaffiliations(name)
		with open('update_googl_big_job.p','wb') as f:
			pickle.dump(gad,f)

try:
	with open('big_g_locations.p','wb') as f:
		g_locations = pickle.load(f)
except:
	g_locations = {}
	for name,institution in gad.items():
		if institution is not None:
			#print(institution,'in google locations')
			search = institution.split(' ')
			if len(search)==3:
				xy = coords_of_target_university(search[-3]+str(' ')+search[-2]+str(' ')+search[-1])
			if len(search)==2:
				xy = coords_of_target_university(search[-2]+str(' ')+search[-1])
			if len(search)==1:
				xy = coords_of_target_university(search[-1])
			if 'xy' in locals():
				check_none = xy[1]
			else:
				check_none = None

			if check_none is not None:
				g_locations[name] = xy
			else:
				if ',' in institution:
					location_key = institution.split(',')[-1]
					xy = coords_of_target_university(location_key)
					check_none = xy[1]
					if check_none is not None:
						g_locations[name] = xy
				else:
					xy = coords_of_target_university(institution)
					check_none = xy[1]
					if check_none is not None:
						g_locations[name] = xy

			if name in g_locations.keys():
				print(name,g_locations[name])
			else:
				print(name,'still not found')

			with open('big_g_locations.p','wb') as f:
				pickle.dump(g_locations,f)

with open('mega_net.p','rb') as f:
	mg = pickle.load(f)

hindex= {}
for name in mg.nodes:

	hindex[name] = author_to_gh_index(name)
	with open('hindex_g_locations.p','wb') as f:
		pickle.dump(hindex,f)
