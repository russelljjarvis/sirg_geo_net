import shelve
import pickle
import networkx

import holoviews as hv
import chord2

hv.extension("bokeh", logo=False)
hv.output(size=100)
MAIN_AUTHOR = "Brian H Smith"
with shelve.open("fast_graphs_splash.p") as db:
    print(list(db.keys()))
"""

with open("Brian H Smith.p","rb") as f:
	g = pickle.load(f)
with open('Brian H Smith_df.p',"rb") as f:
	df = pickle.load(f)


edges_df = networkx.to_pandas_adjacency(g)
fig = chord2.make_filled_chord(edges_df)


graph = hv.Graph.from_networkx(g,networkx.layout.fruchterman_reingold_layout)
graph.opts(color_index='circle', width=350, height=350, show_frame=False,
				 xaxis=None, yaxis=None,tools=['hover','tap'],node_size=10,
				 cmap = ['blue','orange'])

with shelve.open('fast_graphs_splash.p') as db:
	if not 'Brian H Smith' in db.keys():
		db['Brian H Smith'] = {'chord':fig,'graph':graph,'df':df}

with shelve.open('fast_graphs_splash.p') as db:
	print(db['Brian H Smith'])
"""
