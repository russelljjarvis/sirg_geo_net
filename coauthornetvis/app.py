"""
Author: [Russell Jarvis](https://github.com/russelljjarvis)
"""


import shelve
import streamlit as st
import os
import pandas as pd
import pickle
import streamlit as st
from holoviews import opts, dim
from collections import Iterable
import networkx

from auxillary_methods import author_to_coauthor_network, network#,try_again
import holoviews as hv
from auxillary_methods import push_frame_to_screen, plotly_sized#, data_shade, draw_wstate_tree
import chord2
import shelve


def streamlit_maps():
	data = pd.DataFrame({
	'awesome cities' : ['Chicago', 'Minneapolis', 'Louisville', 'Topeka'],
	'latitude' : [41.868171, 44.979840,  38.257972, 39.030575],
	'longitude' : [-87.667458, -93.272474, -85.765187,  -95.702548]
	})


	st.map(data)
def user_manual_fix_missing(list_of_dicts):
	st.sidebar.title("Add new or replace author location in dataframe")
	name = st.sidebar.text_input("Enter Author Name")
	address = st.sidebar.text_input("Insitution Address")
	longitude = st.sidebar.text_input("longitude")
	latitude = st.sidebar.text_input("latitude")

	if st.button("Add row"):
		list_of_dicts.append({"name": name, "address": address, "longitude": longitude, "latitude": latitude})

	st.write(pd.DataFrame(get_data()))
	st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)

def disable_logo(plot, element):
	plot.state.toolbar.logo = None


hv.extension("bokeh", logo=False)
hv.output(size=300)
hv.plotting.bokeh.ElementPlot.finalize_hooks.append(disable_logo)
#"Antonin Delpeuch",

import geopandas
import plotly.graph_objects as go
import pandas as pd
import geopandas
import streamlit as st
import numpy as np
import pickle

from netgeovis2 import main_plot_routine, identify_find_missing,remove_missing_persons_from_big_net
import pandas as pd
with open('both_sets_locations.p','rb') as f:
	both_sets_locations = pickle.load(f)

sirg_author_list = [
	"Brian H Smith",
	"Christian Rabeling",
	"Jon F Harrison",
	"Juergen Liebig",
	"Stephen C Pratt",
	"Jennifer H Fewell",
	"Arianne J Cease",
	"Gro V Amdam"
]

def get_table_download_link_csv(df):
	csv = df.to_csv().encode()
	b64 = base64.b64encode(csv).decode()
	href = f'<a href="data:file/csv;base64,{b64}" download="authors.csv" target="_blank">Download csv file</a>'
	return href

#push_frame_to_screen(df.head())
def passed():
	with shelve.open("fast_graphs_splash.p") as db:
		flag = author_name in db
		if flag:
			try:
				fig_pln = plotly_sized(db[author_name]["fig_pln"])
			except:
				g, df = author_to_coauthor_network(author_name)
				fig_pln = plotly_sized(g)
			db[author_name]["g"] = g
			db[author_name]["fig_pln"] = fig_pln
	st.markdown("""--------------""")
	st.markdown(
		"<h3 style='text-align: left; color: black;'>"
		+ str("Experimental Graph:")
		+ "</h3>",
		unsafe_allow_html=True,
	)
	# st.write(fig_shade)
	st.write(fig_pln)
def big_plot_job():
	if os.path.exists('missing_person.p'):
		with open('missing_person.p','rb') as f:
			temp = pickle.load(f)
		[both_sets_locations,missing_person_name,missing_person_location,both_sets_locations_missing] = temp

	else:
		#except:
		both_sets_locations,missing_person_name,missing_person_location,both_sets_locations_missing = identify_find_missing()
		temp = [both_sets_locations,missing_person_name,missing_person_location,both_sets_locations_missing]
		with open('missing_person.p','wb') as f:
			pickle.dump(temp,f)

	node_positions = list(both_sets_locations.values())
	long_lat = [np[1] for np in node_positions if np[1] is not None]
	lat = [ coord[0] for coord in long_lat]
	long = [ coord[1] for coord in long_lat]
	node_location_name = [np[0] for np in node_positions if np[1] is not None]

	node_person = list([k for k,v in both_sets_locations.items() if v[0] is not None])
	#if False:
	if os.path.exists('big_g_locations.p'):
		try:
			with open('big_g_locations.p','rb') as f:
				g_locations = pickle.load(f)
			both_sets_locations.update(g_locations)
			missing_person_name = list(set(missing_person_name) -set(g_locations.keys()))

		except:
			pass
	plt_unbundled,plt_bundled,ax3 = main_plot_routine(both_sets_locations,missing_person_name,node_location_name)#,author_list)

def main():
	st.markdown("""--------------""")
	st.title("""Geo Geographic Maps are hard to compute, so first we will populate the screen
	with other stuff while we build them...""")




	figure_size = 200
	hv.output(size=figure_size)
	'''
	st.title("Create a Coauthorship Network")
	author_name1 = st.text_input("Enter Author Name:")

	Note: Search applies [dissmin](https://dissemin.readthedocs.io/en/latest/api.html) API backend

	options = tuple(sirg_author_list)
	author_name0 = st.sidebar.radio("Which author are you interested in?", options)

	if author_name1:
		author_name = author_name1
		author_name0 = None

	if author_name0:
		author_name = author_name0
		author_name1 = None

	if author_name0 or author_name1:
		st.markdown("""## You Chose the Author: {0} """.format(author_name))
		MAIN_AUTHOR = author_name
	'''
	with shelve.open("fast_graphs_splash.p") as db:
		flag = author_name in db
		if False:
		#if flag:
			fig = db[author_name]["chord"]
			graph = db[author_name]["graph"]
			df = db[author_name]["df"]
			if "fig_pln" in db[author_name].keys():
				fig_pln = db[author_name]["fig_pln"]
			if "g" in db[author_name].keys():
				g = db[author_name]["g"]

		if flag:
			g, df = author_to_coauthor_network(author_name)
			fig_pln = plotly_sized(g)
			graph = hv.Graph.from_networkx(
				g, networkx.layout.fruchterman_reingold_layout
			)
			graph.opts(
				color_index="circle",
				width=450,
				height=450,
				show_frame=False,
				xaxis=None,
				yaxis=None,
				tools=["hover", "tap"],
				node_size=10,
				cmap=["blue", "orange"],
			)
			# plot=dict(finalize_hooks=[disable_logo]),
			edges_df = networkx.to_pandas_adjacency(g)
			fig = chord2.make_filled_chord(edges_df)
			db[author_name] = {
				"chord": fig,
				"graph": graph,
				"df": df,
				"fig_pln": fig_pln,
				"g": g,
			}
		if len(db.keys())>20:
			for k in db.keys():
				try:
					print('try to thin out the dictionary')
					db.pop(k,None)
				except:
					pass
		db.close()

	label = "Coauthorship Chord Network for: " + MAIN_AUTHOR
	st.markdown(
		"<h3 style='text-align: left; color: black;'>" + label + "</h3>",
		unsafe_allow_html=True,
	)
	st.write(fig)
	st.markdown("""--------------""")

	label = "Coauthorship Network for: " + MAIN_AUTHOR
	st.markdown(
		"<h3 style='text-align: left; color: black;'>" + label + "</h3>",
		unsafe_allow_html=True,
	)
	st.write(hv.render(graph, backend="bokeh"))

	#st.markdown("""Geo Geographic Maps computing now, this will take time""")
	st.markdown(
		"<h1 style='text-align: left; color: black;'>"
		+ str("Geo Geographic Maps computing now, this will take time")
		+ "</h1>",
		unsafe_allow_html=True,
	)
	'''
	A lot of potential coauthors were excluded
	see this list below:
	'''
	missing_person_name = list([k for k,v in both_sets_locations.items() if v[1] is None])
	with open('missing_person_name.p','rb') as f:
		missing_person_name = pickle.load(f)
	ds_nodes = pd.DataFrame(missing_person_name, columns=["names"])
	st.dataframe(ds_nodes)

	big_plot_job()
	st.markdown(
		"<h3 style='text-align: left; color: black;'>"
		+ str("Here are some of the publications dissmin used to build these networks.")
		+ "</h3>",
		unsafe_allow_html=True,
	)
	push_frame_to_screen(df)



	st.markdown("""[My other science information dashboard app](https://agile-reaches-20338.herokuapp.com/)""")
	"""
	[Source Code:](https://github.com/russelljjarvis/CoauthorNetVis)
	"""


	#st.markdown("""## Graphs of Entire SIRG network """)#.format(author_name))
	#st.markdown(""" This will take a long time """)#.format(author_name))


	#ego_graph(mg)
	#fb_graph = try_again(mg)
	#fig_pln = plotly_sized(mg)
	#st.write(hv.render(fb_graph, backend="bokeh"))
	#edges_df_full = networkx.to_pandas_adjacency(mg)
	#st.markdown("""## ----------------- """)#.format(author_name))

	#fig = chord2.make_filled_chord(edges_df_full)
	#st.write(fig)
	#st.markdown("""## ----------------- """)#.format(author_name))
	#try:
	#    from PIL import Image
	#    image = Image.open("whole_net_not_dot_huge.png")
	#    st.image(image, caption='Whole SIRG Network',
	#            use_column_width=True)
	#except:
	#    pass

if __name__ == "__main__":
	main()
	# from multiapp import MultiApp
	# from apps import home, data_stats # import your app modules here

	# app = MultiApp()

	# Add all your application here
	# app.add_app("Home", home.app)
	# app.add_app("Data Stats", data_stats.app)

	# The main app
	# app.run()
