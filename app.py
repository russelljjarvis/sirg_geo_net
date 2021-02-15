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

from auxillary_methods import author_to_coauthor_network,network
import holoviews as hv
import shelve
from auxillary_methods import push_frame_to_screen

hv.extension('bokeh', logo=False)
hv.output(size=100)
author_list = [None,
"Brian H Smith",
"Christian Rabeling",
"Jon F Harrison",
"Juergen Liebig",
"Stephen C Pratt",
"Jennifer H Fewell",
"Arianne J Cease",
"Gro V Amdam"]


import shelve
#from holoviews.operation.datashader import datashade, bundle_graph

def main():
	MAIN_AUTHOR = "Brian H Smith"
	with open("Brian H Smith.p","rb") as f:
		g = pickle.load(f)
	with open('Brian H Smith_df.p',"rb") as f:
		df = pickle.load(f)

	options = [150,175,200,50,75,100,125]
	figure_size = st.sidebar.radio("Figure size (smaller-faster)",options)
	hv.output(size=figure_size)

	st.title('Create Coauthorship Network of Science Author')
	author_name1 = st.text_input('Enter Author Name:')

	options = tuple(author_list)
	author_name0 = st.sidebar.radio("Which SIRG author are you interested in?",options)

	if author_name1:
		author_name = author_name1
		author_name0 = None
		author_list.extend(author_name1)
	if author_name0:
		author_name = author_name0
		author_name1 = None

	if author_name0 or author_name1:
		st.markdown("""## You Chose the Author: {0} """.format(author_name))
		g,df = author_to_coauthor_network(author_name)
		#if g is None:
		#	del g
		MAIN_AUTHOR = author_name
		if str("graph") in locals():
			del graph
			del chord

	label="Coauthorship Network for: "+MAIN_AUTHOR
	st.markdown(label)


	#pos = nx.spring_layout(g, seed=0)
	#nx.draw_networkx(g, pos)

	#for edge in g.edges(data='weight'):
	#    nx.draw_networkx_edges(g, pos, edgelist=[edge], width=edge[2])
	graph = hv.Graph.from_networkx(g,networkx.layout.fruchterman_reingold_layout)
	graph.opts(color_index='circle', width=350, height=350, show_frame=False,
					 xaxis=None, yaxis=None,tools=['hover','tap'],node_size=10,
					 cmap = ['blue','orange'])
	st.write(hv.render(graph, backend='bokeh'))
	#edge_list = networkx.to_edgelist(g)
	#label="Coauthorship Network for: "+MAIN_AUTHOR
	#chord = hv.Chord(edge_list)#label=label)#.opts(tools=['hover','tap'],node_size=10,
	#node_color='type',
 	#cmap = ['blue','orange'])
	#chord.opts(color_index='circle', width=350, height=350, show_frame=False,
	#				 xaxis=None, yaxis=None)#,label=label)
	label="Coauthorship Chord Network for: "+MAIN_AUTHOR
	st.markdown(label)

	#st.write(hv.render(chord, backend='bokeh'))
	#st.text(dir(g))
	###
	import chord2
	#edges_df = networkx.to_pandas_edgelist(g)
	edges_df = networkx.to_pandas_adjacency(g)
	fig = chord2.make_filled_chord(edges_df)
	st.write(fig)
	###
	if 'df' in locals():
		st.markdown(""" ### Here are some of the publications we are using to build the networks. """)
		push_frame_to_screen(df)

	#edges_df = networkx.to_pandas_edgelist(g)
	#fb_nodes = hv.Nodes(g.nodes)#.sort()
	#fb_graph = hv.Graph((edges_df, fb_nodes), label=label)
	#colors = ['#000000']+hv.Cycle('Category20').values

	#fb_graph.opts(cmap=colors, node_size=10, edge_line_width=1)
	#              node_line_color='gray', node_color='circle')
	#try:
	#	bundled = bundle_graph(fb_graph)
	#		st.write(hv.render(bundled, backend='bokeh'))
	#except:
	#	pass
	#from pandapower.plotting.plotly import simple_plotly
	#from pandapower.networks import mv_oberrhein
	#net = mv_oberrhein()

	#st.write(simple_plotly(net, on_map=True, projection='epsg:31467'))
		#datashade(bundle_graph(fb_graph), width=800, height=800) *\
		#bundled.select(circle=MAIN_AUTHOR).opts(node_fill_color='white')
		#st.write(hv.render(bundled, backend='bokeh'))
	#sankey = hv.Sankey(data_frame)
	#sankey.opts(color_index='circle', width=400, height=400, show_frame=False,
	#				 xaxis=None, yaxis=None)
	#st.write(hv.render(sankey, backend='bokeh'))
	#graph = hv.Graph.from_networkx(g,networkx.layout.spring_layout)
	#graph.opts(color_index='circle', width=350, height=350, show_frame=False,
	#				 xaxis=None, yaxis=None,tools=['hover','tap'],node_size=10,
	#				 cmap = ['blue','orange'])
	#st.write(hv.render(graph, backend='bokeh'))


if __name__ == "__main__":
	main()
