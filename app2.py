"""
## Science Readability Project

To ensure that writing is accessible to the general population, authors must consider the length of written text, as well as sentence structure, vocabulary, and other language features. While popular magazines, newspapers, and other outlets purposefully cater language for a wide audience, there is a tendency for academic writing to use more complex, jargon-heavy language.

In the age of growing science communication, this tendency for scientists to use more complex language can carry over when writing in more mainstream media, such as blogs and social media. This can make public-facing material difficult to comprehend, undermining efforts to communicate scientific topics to the general public. While readability tools, such as Readable and Upgoer5 currently exist to report on readability of text, they report the complexity of only a single document. In addition, these tools do not focus on complexity in a more academic-type context.

To address this, we created a tool that uses a data-driven approach to provide authors with insights into the readability of the entirety of their published scholarly work with regard to other text repositories. The tool first quantifies an existing text repository [@Soldatova:2007] with complexity shown to be comparable to that of other scientific journals. The tool subsequently uses this output as a reference to show how the readability of user-selected written work compares to this source.

Ultimately, this tool will expand upon current readability metrics by computing a more detailed and comparative look at the complexity of written text. We hope that this will allow scientists and other experts to better monitor the complexity of their writing relative to other text types, leading to the creation of more accessible online material. And perhaps more broadly contribute to an improved global communication and understanding of complex topics.

Author: [Russell Jarvis](https://github.com/russelljjarvis)\n
Author: [Patrick McGurrin](https://github.com/mcgurrgurr)\n
Source: [Github](https://github.com/russelljjarvis/ScienceAccess)
"""

import streamlit as st
import os
import pandas as pd
import pickle
import numpy as np
#import plotly.figure_factory as ff
#import plotly.express as px
import copy
import streamlit as st
import math
import scipy
#import plotly.graph_objects as go


from auxillary_methods import author_to_coauthor_network,network
import holoviews as hv

from holoviews import opts, dim
from collections import Iterable
import networkx
hv.extension('bokeh', logo=False)

author_list = [
"Brian H Smith",
"Christian Rabeling",
"Jon Harrison",
"Juergen Liebig",
"Stephen C Pratt",
"Jennifer H Fewell",
"Arianne Cease"]
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
