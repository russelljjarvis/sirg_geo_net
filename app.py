"""
Author: [Russell Jarvis](https://github.com/russelljjarvis)\n
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

from auxillary_methods import author_to_coauthor_network, network
import holoviews as hv
import shelve
from auxillary_methods import push_frame_to_screen
import chord2
import shelve

hv.extension("bokeh", logo=False)
hv.output(size=100)
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

def main():
	MAIN_AUTHOR = "Brian H Smith"

	#options = [150, 175, 200, 50, 75, 100, 125]
	#figure_size = st.sidebar.radio("Figure size (smaller-faster)", options)
	figure_size = 150
	hv.output(size=figure_size)

	st.title("Create Coauthorship Network of Science Author")
	author_name1 = st.text_input("Enter Author Name:")

	options = tuple(author_list)
	author_name0 = st.sidebar.radio("Which SIRG author are you interested in?", options)

	if author_name1:
		author_name = author_name1
		author_name0 = None
		author_list.insert(0,author_name1)

	if author_name0:
		author_name = author_name0
		author_name1 = None

	if author_name0 or author_name1:
		st.markdown("""## You Chose the Author: {0} """.format(author_name))
		MAIN_AUTHOR = author_name

	with shelve.open('fast_graphs_splash.p') as db:
		flag = author_name in db
		if flag:
			fig = db[author_name]['chord']#:fig,'graph':graph}
			graph = db[author_name]['graph']
			#if 'df' in db[author_name].keys():
			df = db[author_name]['df']
			#else:
			#	g, df = author_to_coauthor_network(author_name)
			#	db[author_name] = {'chord':fig,'graph':graph,'df':df}

		if not flag:
			@st.cache(suppress_st_warning=True)
			def wrapper(author_name):
				g, df = author_to_coauthor_network(author_name)
				return g,df
			#g, df = author_to_coauthor_network(author_name)
			#g,df = wrapper(author_name)
			graph = hv.Graph.from_networkx(g, networkx.layout.fruchterman_reingold_layout)
			graph.opts(
				color_index="circle",
				width=350,
				height=350,
				show_frame=False,
				xaxis=None,
				yaxis=None,
				tools=["hover", "tap"],
				node_size=10,
				cmap=["blue", "orange"],
			)
			edges_df = networkx.to_pandas_adjacency(g)
			fig = chord2.make_filled_chord(edges_df)
			db[author_name] = {'chord':fig,'graph':graph,'df':df}
		db.close()
	label = "Coauthorship Chord Network for: " + MAIN_AUTHOR
	st.markdown(
		"<h3 style='text-align: left; color: black;'>" + label + "</h3>",
		unsafe_allow_html=True,
	)
	st.write(fig)
	st.markdown("""----""")

	label = "Coauthorship Network for: " + MAIN_AUTHOR
	st.markdown(
		"<h3 style='text-align: left; color: black;'>" + label + "</h3>",
		unsafe_allow_html=True,
	)
	st.write(hv.render(graph, backend="bokeh"))

	###
	#if "df" in locals():
	st.markdown(
			"<h3 style='text-align: left; color: black;'>"
			+ str(
				"Here are some of the publications we are using to build the networks."
			)
			+ "</h3>",
			unsafe_allow_html=True,
	)
	push_frame_to_screen(df)


if __name__ == "__main__":
	main()
	# from multiapp import MultiApp
	# from apps import home, data_stats # import your app modules here

	# app = MultiApp()

	# Add all your application here
	# app.add_app("Home", home.app)
	# app.add_app("Data Stats", data_stats.app)

	# The main app
	#app.run()
