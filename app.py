"""

Author: [Russell Jarvis](https://github.com/russelljjarvis)\n
"""

import streamlit as st
import os
import pandas as pd
import pickle
import streamlit as st

from holoviews import opts, dim
from collections import Iterable
import networkx

from auxillary_methods import author_to_coauthor_network,network
import holoviews as hv
import shelve
from auxillary_methods import push_frame_to_screen

#hv.extension('bokeh')
#hv.output(size=200)
hv.extension('bokeh', logo=False)
hv.output(size=100)
author_list = [None,
"Brian H Smith",
"Christian C Rabeling",
"Jon F Harrison",
"Juergen Liebig",
"Stephen C Pratt",
"Jennifer H Fewell",
"Arianne J Cease",
"Gro V Amdam"]



def main():
	MAIN_AUTHOR = "Brian H Smith"
	with open("Brian H Smith.p","rb") as f:
		g = pickle.load(f)
	options = [100,125,150,175,200,50,75]
	figure_size = st.sidebar.radio("Figure size (smaller-faster)",options)
	hv.output(size=figure_size)

	st.title('Create Coauthorship Network of Science Author')
	author_name1 = st.text_input('Enter Author Name:')

	options = tuple(author_list)
	author_name0 = st.sidebar.radio("Which SIRG author are you interested in?",options)

	if author_name1:
		author_name = author_name1
		author_name0 = 0
		author_list.extend(author_name1)
	if author_name0:
		author_name = author_name0
		author_name1 = 0

	if author_name0 or author_name1:
		st.markdown("""## You Chose the Author: {0} """.format(author_name))
		g,df = author_to_coauthor_network(author_name)
		MAIN_AUTHOR = author_name
		if str("graph") in locals():
			del graph
			del chord

	label="Coauthorship Network for: "+MAIN_AUTHOR
	st.markdown(label)

	graph = hv.Graph.from_networkx(g,networkx.layout.fruchterman_reingold_layout)
	graph.opts(color_index='circle', width=350, height=350, show_frame=False,
					 xaxis=None, yaxis=None)
	st.write(hv.render(graph, backend='bokeh'))
	edge_list = networkx.to_edgelist(g)
	#label="Coauthorship Network for: "+MAIN_AUTHOR
	chord = hv.Chord(edge_list,label=label)
	graph.opts(color_index='circle', width=400, height=400, show_frame=False,
					 xaxis=None, yaxis=None)
	st.write(hv.render(chord, backend='bokeh'))

	if 'df' in locals():
		st.markdown(""" ### Here are some of the publications we are using to build the networks. """)
		push_frame_to_screen(df)


if __name__ == "__main__":
	main()
