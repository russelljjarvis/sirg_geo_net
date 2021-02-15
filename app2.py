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

hv.extension('bokeh', logo=False)

author_list = [
"Brian H Smith",
"Christian C Rabeling",
"Jon F Harrison",
"Juergen Liebig",
"Stephen C Pratt",
"Jennifer H Fewell",
"Arianne J Cease"]
def main():

	st.title('Create Coauthorship Network of Science Author')
	author_name = st.text_input('Enter Author Name:')

	author_name = st.sidebar.multiselect("Which SIRG author are you interested in?"
	  ,tuple(author_list))

	if author_name:
		"""## You Chose the Author: """
		if len(author_name[0])>1:
			author_name = author_name[0]
		st.text(author_name)

		try:

			coauthors = author_to_coauthor_network(author_name)
			g = network(coauthors,author_name)
			MAIN_AUTHOR = author_name
		except:
			coauthors = author_to_coauthor_network(author_name)
			g = network(coauthors,author_name)
			MAIN_AUTHOR = author_name

	else:
		""" ## Splash screen only, waiting for input... """
		""" ### Coauthorship Networks Brian Smith... """

		with open("Brian H Smith.p","rb") as f:
			g = pickle.load(f)
		MAIN_AUTHOR = "Brian H Smith"
	hv.extension('bokeh')
	hv.output(size=200)

	graph = hv.Graph.from_networkx(g,networkx.layout.fruchterman_reingold_layout)
	#colors = ['#000000']+hv.Cycle('Category20').values
	graph.opts(color_index='circle', width=350, height=350, show_frame=False,
					 xaxis=None, yaxis=None) #node_size=25, edge_line_width=1
	st.write(hv.render(graph, backend='bokeh'))

	edge_list = networkx.to_edgelist(g)
	try:
		label="Coauthorship Network for: "+MAIN_AUTHOR
		chord = hv.Chord(edge_list,label=label)

	except:
		chord = hv.Chord(edge_list)
	st.write(hv.render(chord, backend='bokeh'))


if __name__ == "__main__":
	main()
