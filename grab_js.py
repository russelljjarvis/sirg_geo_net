
import js2py
import pandas as pd
import pickle
import glob
import requests
from flask import render_template
import json
from geopy.geocoders import Nominatim

#results = js2py.run_file("data/affiliationList.js")
#affil_dict = results[0].to_dict()



jsdatafiles = [f for f in glob.glob("data/js/*.js")]
js_data_contents = {}
for f in jsdatafiles:
	results = js2py.run_file(f)
	key = str(f.split("data/js/")[1])
	js_data_contents[key] = results[0].to_dict()

jsondatafiles = [f for f in glob.glob("data/js/*.json")]
json_data_contents = {}
for name in jsondatafiles:
	with open(name,'rb') as f:
		results = json.load(f)
	key = str(name.split("data/js/")[1])
	json_data_contents[key] = results#results[0].to_dict()

global py_geo_map
py_geo_map = {}
results = js2py.run_file("data/affiliationList.js")
holding = results[0].to_dict()
#pdb.set_trace()
for k,v in holding.items():
	key = v['Name']
	value = v['Position']
	py_geo_map[key] = value

def university_data_frame():
	wu = pd.read_csv("data/world-universities.csv")
	wu.rename(
		columns={
			"AD": "country",
			"University of Andorra": "university",
			"http://www.uda.ad/": "wesbite",
		},
		inplace=True,
	)
	return wu, list(wu["university"])


def coords_of_target_university(search_key):
	# search_key = "Arizona State"
	# search_key = "University of Melbourne"

	for k,v in py_geo_map.items():
		if search_key in k:
			return (k,v)
		else:

			geolocator = Nominatim(user_agent="SIRG")
			location = geolocator.geocode(search_key)
			if location is not None:
				v = (location.latitude,location.longitude)
				print(search_key,v)
				return (search_key, v)
			else:
				print(None,search_key)

		#else:
		#    if k in search_key:
		#        return (k,v)

	return None

def update_csv_with_website_lang_long():
	wu, list_univ = university_data_frame()#.to_dict()

	for un in list_univ:
		key = un
		ret = coords_of_target_university(key)
		if ret is not None:
			name,coords = ret
			# add a new coordinates column to data frame.
			wu[un]["coords"] = coords

#def unpickle_citations(author_name,results):
#    author_name = str(author_name)
#    with open('scholar_results'+str(author_name)+'.p','rb') as f:
#        results = pickle.load(f)

def sirg_author_locations():
	name_to_place = {}
	name_to_affil = {}
	re = [f for f in glob.glob("data/scholar_results*.p")]
	for i in re:
		with open(i,'rb') as f:
			temp_results = pickle.load(f)
			for author_name,value in temp_results.items():
				university = value['affiliation'].split(", ")
				name_to_affil[author_name] = university
				if len(university)>1:
					query = university[1]
					ret = coords_of_target_university(query)
				else:
					query = value['affiliation']
					ret = coords_of_target_university(query)
				if ret is not None:
					name,coords = ret
					name_to_place[author_name] = coords
				else:
					#coords = query_google(query)
					if coords in locals():
						name_to_place[author_name] = coords
					else:
						name_to_place[author_name] = None
	return name_to_place,name_to_affil
"""
def query_google(location):
	URL = "https://geocode.search.hereapi.com/v1/geocode"
	#location = #input("Enter the location here: ") #taking user input
	#api_key = 'YOUR_API_KEY' # Acquire from developer.here.com
	PARAMS = {'apikey':"AIzaSyBMPpDa8yovQGk-A5XMMyrhzwvIaylZ9Wo",'q':location}
	# sending get request and saving the response as response object
	r = requests.get(url = URL, params = PARAMS)
	data = r.json()
	print(data)
	latitude = data['items'][0]['position']['lat']
	longitude = data['items'][0]['position']['lng']
	return (longitude,latitude)
"""
##
# Now write this backinto a file.
##
