#!/usr/bin/env python
# coding: utf-8
# Import libraries
import pandas as pd
import numpy as np
import math

import geopandas
import json

from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.palettes import brewer
from bokeh.io.doc import curdoc
from bokeh.models import Slider, HoverTool, Select
from bokeh.layouts import widgetbox, row, column

import seaborn as sns

import geopandas
import plotly.graph_objects as go
import pandas as pd
import geopandas
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from auxillary_methods import author_to_coauthor_network, network
import colorlover as cl
from matplotlib import cm
import holoviews as hv
from auxillary_methods import plotly_sized
from grab_js import coords_of_target_university  # , university_data_frame

from collections import OrderedDict
from datashader.bundling import hammer_bundle

import pickle
import streamlit as st
import os
import plotly.express as px
from auxillary_methods import tqdm

import pylab as plot

def rectify_data_sources():
    with open("both_sets_locations.p", "rb") as f:
        both_sets_locations = pickle.load(f)
    both_sets_locations["Chelsea N. Cook"] = both_sets_locations["Brian H. Smith"]
    both_sets_locations["Hong Lei"] = both_sets_locations["Brian H. Smith"]
    both_sets_locations["Richard C. Gerkin"] = both_sets_locations["Brian H. Smith"]
    both_sets_locations["Teddy Cogley"] = both_sets_locations["Brian H. Smith"]

    sirg_author_list = [
        "Brian H Smith",
        "Christian Rabeling",
        "Jon F Harrison",
        "Juergen Liebig",
        "Stephen C Pratt",
        "Jennifer H Fewell",
        "Arianne J Cease",
        "Gro V Amdam",
    ]

    for al in sirg_author_list:
        both_sets_locations[al] = both_sets_locations["Brian H. Smith"]

    node_person = list(both_sets_locations.keys())

    with open("mega_net.p", "rb") as f:
        mg = pickle.load(f)

    return mg, node_person, both_sets_locations, sirg_author_list


def identify_find_missing():
    mg, node_person, both_sets_locations, sirg_author_list = rectify_data_sources()
    both_sets_locations_complete = {
        k: v for k, v in both_sets_locations.items() if v[1] is not None
    }

    node_person = list([k for k, v in both_sets_locations.items() if v[1] is not None])
    missing_person_name = list(
        [k for k, v in both_sets_locations.items() if v[1] is None]
    )
    missing_person_location = list(
        [v for k, v in both_sets_locations.items() if v[1] is None]
    )
    both_sets_locations_missing = {
        k: v for k, v in both_sets_locations.items() if v[1] is None
    }
    list_of_dicts = [v for k, v in both_sets_locations.items()]
    df = pd.DataFrame(list_of_dicts)

    ###
    # st.dataframe(df)
    ###
    if os.path.exists("retry.p"):
        with open("retry.p", "rb") as f:
            retry = pickle.load(f)
    else:
        retry = {}

        for person_key, v in both_sets_locations_missing.items():
            location_key = both_sets_locations_missing[person_key][0].split(",")[-1]
            ##
            # More aggressively try different variations of affiliation address strings.
            #
            ##
            xy = coords_of_target_university(location_key)
            # check_none = xy#[1]
            if xy is None:
                search = both_sets_locations_missing[person_key][0].split(
                    " "
                )  # [-2:-1]
                xy = coords_of_target_university(search[-2] + str(" ") + search[-1])
                if xy is not None:
                    # check_none = xy[1]
                    retry[person_key] = xy
            else:
                retry[person_key] = xy
                check_none = xy[1]
        assert retry is not None
        # import pprint
        # pprint.pprint(retry)

        retry = {k: v for k, v in retry.items() if v[1] is not None}
        both_sets_locations_complete.update(retry)
        list_of_dicts = [v for k, v in both_sets_locations_missing.items()]
        df = pd.DataFrame(list_of_dicts)
        ###
        # st.dataframe(df)
        ###
        import copy

        with open("retry.p", "wb") as f:
            retry = pickle.dump(copy.copy(retry), f)
        # assert retry is not None

    try:
        with open("big_g_locations.p", "rb") as f:
            g_locations = pickle.load(f)
        both_sets_locations_complete.update(g_locations)
    except:
        pass
    # list_of_dicts = [ v for k,v in both_sets_locations.items()]
    # df = pd.DataFrame(list_of_dicts)
    # st.dataframe(df)

    both_sets_locations = both_sets_locations_complete
    missing_person_name = list(
        [k for k, v in both_sets_locations.items() if v[1] is None]
    )
    missing_person_location = list(
        [v for k, v in both_sets_locations.items() if v[1] is None]
    )
    both_sets_locations_missing = {
        k: v for k, v in both_sets_locations.items() if v[1] is None
    }
    # list_of_dicts = [ v for k,v in both_sets_locations_missing.items()]
    # df = pd.DataFrame(list_of_dicts)
    # st.dataframe(df)
    return (
        mg,
        both_sets_locations,
        missing_person_name,
        missing_person_location,
        both_sets_locations_missing,
        sirg_author_list,
    )
    # return mg,both_sets_locations,missing_person_name,missing_person_location,both_sets_locations_missing


# def
'''
def test():
    """
    not used
    """
    import pandas as pd
    import sys
    from bokeh.io import output_notebook, show, output_file
    from bokeh.plotting import figure
    from bokeh.models import GeoJSONDataSource, ColumnDataSource

    # Input GeoJSON source that contains features for plotting.
    geosource = GeoJSONDataSource(geojson=grid)
    pointsource = ColumnDataSource(datapoints)

    # Create figure object.
    p = figure(
        title="Geographic distribution of 2019-nCov cases worldwide",
        plot_height=600,
        plot_width=950,
    )
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    # Add patch renderer to figure.
    p.patches(
        "xs",
        "ys",
        source=geosource,
        fill_color="lightgrey",
        line_color="black",
        line_width=0.25,
        fill_alpha=1,
    )

    p.circle("lat", "lon", source=pointsource, size=15)
    # Display figure inline in Jupyter Notebook.
    # output_notebook()
    # Display figure.
    show(p)
'''


def remove_missing_persons_from_big_net(both_sets_locations, missing_person_name):
    (
        mg,
        both_sets_locations,
        missing_person_name,
        missing_person_location,
        both_sets_locations_missing,
        sirg_author_list,
    ) = identify_find_missing()

    node_positions = list(both_sets_locations.values())
    node_location_name = [np[0] for np in node_positions if np[1] is not None]
    node_positions = list(both_sets_locations.values())
    long_lat = [np[1] for np in node_positions if np[1] is not None]
    lat = [coord[0] for coord in long_lat]
    long = [coord[1] for coord in long_lat]

    G = mg
    positions = {}
    second = nx.DiGraph()
    for node in both_sets_locations.keys():
        if node in missing_person_name:
            if node in G.nodes:
                G.remove_node(node)
        else:
            if not node in both_sets_locations.keys():
                if node in G.nodes:
                    G.remove_node(node)
                    missing_person.append(node)

    for node in both_sets_locations.keys():
        if node in set(both_sets_locations.keys()) and node in set(G.nodes):
            lat_long = both_sets_locations[node]

            second.add_node(node, pos=(lat_long[1][1], lat_long[1][0]))
            positions[node] = lat_long[1]
            G.nodes[node]["pos"] = lat_long[1]
    for node in second.nodes:
        for ne in G.out_edges(node):
            if ne[0] in second.nodes and ne[1] in second.nodes:
                second.add_edge(node, ne[1])

    return G, second, lat, long, node_location_name, sirg_author_list


def holoviews_bundle(mg):
    """
    Note this would work if I use other edge bundling code
    """

    def disable_logo(plot, element):
        plot.state.toolbar.logo = None

    hv.extension("bokeh", logo=False)
    hv.output(size=300)
    hv.plotting.bokeh.ElementPlot.finalize_hooks.append(disable_logo)

    fb_nodes = hv.Nodes(mg.nodes).sort()
    edges_df = pd.DataFrame(mg.edges, columns=["source", "target"])
    fb_graph = hv.Graph((edges_df, fb_nodes))

    fb_graph.opts(
        cmap=colors,
        tools=["hover", "tap"],
        width=450,
        height=450,
        node_size=10,
        edge_line_width=1,
        node_line_color="gray",
        node_color="circle",
    )


def recalculate_sirg_db(sirg_author_list):
    sn={}
    for k in sirg_author_list:
        g, df = author_to_coauthor_network(k)
        sn[k] = g
    for name in sirg_author_list:
        assert name in sn.keys()
    with open("subnets_pickle.p", "wb") as f:
        pickle.dump(sn, f)

    return sn


def recalculate_sirg_sub_nets(sirg_author_list):
    subnets = recalculate_sirg_db(sirg_author_list)
    #with open("subnets_pickle.p", "rb") as f:
    #    subnets = pickle.load(f)
    for name in sirg_author_list:
        assert name in subnets.keys()
    to_mega_net = list(subnets.values())
    for i, mn in enumerate(to_mega_net):
        if i == 0:
            megnet = mn
        else:
            megnet = nx.compose(megnet, mn)

    re_sirg_author_list = [
        "Brian H. Smith",
        "Christian Rabeling",
        "Jon F. Harrison",
        "Juergen Liebig",
        "Stephen C. Pratt",
        "Jennifer H. Fewell",
        "Arianne J. Cease",
        "Gro V. Amdam",
    ]
    sirg_author_list.extend(re_sirg_author_list)
    #k = second.subgraph(sirg_author_list)
    pos = nx.spring_layout(megnet)  #setting the positions with respect to G, not k.
    plt.figure()
    H_new = nx.DiGraph()
    for sa in sirg_author_list:
        to_mega_net = list(subnets.values())
        for i, mn in enumerate(to_mega_net):
            for edge in mn.edges:
                if edge[0] in sa:
                    if edge[0] not in H_new.nodes:
                        H_new.add_node(edge[0])
                    if edge[1] in sa:
                        H_new.add_edge((edge[0],edge[1]))
                        print((edge[0],edge[1]))

    nx.draw_networkx(H_new, pos=pos)
    plt.show()

    return subnets, megnet

#import networkx as nx
#import networkx as nx
#from matplotlib import pylab as plt

def edge_bundle(graph, world, colors, sirg_author_list, tab10,streamlit=False):
    nodes = graph.nodes
    second = graph
    orig_pos = nx.get_node_attributes(second, "pos")
    pos_ = nx.get_node_attributes(graph, "pos")
    if False:
        #if os.path.exists("segments.p"):
        with open("segments.p", "rb") as f:
            segments = pickle.load(f)

    else:
        nodes_ind = [i for i in range(0, len(graph.nodes()))]
        redo = {k: v for k, v in zip(graph.nodes, nodes_ind)}

        coords = []
        for node in graph.nodes:
            x, y = pos_[node]
            coords.append((x, y))
        nodes_py = [
            [new_name, pos[0], pos[1]]
            for name, pos, new_name in zip(nodes, coords, nodes_ind)
        ]
        ds_nodes = pd.DataFrame(nodes_py, columns=["name", "x", "y"])

        ds_edges_py = []
        for (n0, n1) in graph.edges:
            ds_edges_py.append([redo[n0], redo[n1]])

        ds_edges = pd.DataFrame(ds_edges_py, columns=["source", "target"])
        hb = hammer_bundle(ds_nodes, ds_edges)

        hbnp = hb.to_numpy()
        splits = (np.isnan(hbnp[:, 0])).nonzero()[0]

        start = 0
        segments = []
        for stop in splits:
            seg = hbnp[start:stop, :]
            segments.append(seg)
            start = stop
    fig, ax = plt.subplots(figsize=(40, 40))

    ax = world.plot(color="white", edgecolor="black", figsize=(40, 40))
    # for seg in segments:  # [::100]:
    for ind, seg in enumerate(tqdm(segments, title="Bundling Edges")):
        ax.plot(seg[:, 0], seg[:, 1], linewidth=0.125)
    assert segments is not None

    with open("segments.p", "wb") as f:
        pickle.dump(segments, f)
    ax3 = nx.draw_networkx_nodes(
        graph,
        orig_pos,
        node_size=55,
        node_color=colors,
        node_shape="o",
        alpha=1.0,
        vmin=None,
        vmax=None,
        linewidths=None,
        label=None,
    )  # , **kwds)
    xy = (33.42152185,-111.93316158417922)
    radius = 50
    #ax4 = circle( xy, radius, color="lightsteelblue", facecolor="lightsteelblue", alpha=1, ax=ax3 )
    #xy = (-111.93316158417922,33.42152185,)

    #ax4 = circle( xy, radius, color="lightsteelblue", facecolor="lightsteelblue", alpha=1, ax=ax3 )

    ax5 = plt.scatter(-111.93316158417922, 33.42152185, s=580, facecolors='r', edgecolors='r')
    plt.text(-111.93316158417922, 33.42152185,"SIRG \n Arizona State University",size=25)

    #plt.text(-111.93316158417922, 33.42152185,"ASU")
    for i, v in enumerate(sirg_author_list):
        plt.scatter([], [], c=tab10[i], label="SIRG PI {0}".format(v))
    plt.legend()
    params = {'legend.fontsize': 25}
    plt.rcParams.update(params)
    plot.rcParams.update(params)
    #plt.savefig, bbox_inches="tight")

    plt.savefig("bundled_graph_static.png", bbox_inches="tight")
    if streamlit:
        st.pyplot(plt,use_column_width=False,width=None)



    #G = nx.karate_club_graph()
                                  #just to demonstrate that G.subgraph is okay
                                  #with nodes not in G.

    return fig, ax3, ax3, segments




def edge_bundle_plotly(
    graph, world, colors, sirg_author_list, tab10, segments=None, pos_=None,
    streamlit=False, just_nodes= True
):
    nodes = graph.nodes
    second = graph
    orig_pos = nx.get_node_attributes(second, "pos")
    nodes_ind = [i for i in range(0, len(graph.nodes()))]
    redo = {k: v for k, v in zip(graph.nodes, nodes_ind)}
    if pos_ is None:
        pos_ = nx.get_node_attributes(graph, "pos")

    #assert segments is not None

    #if segments is None:
    coords = []
    for node in graph.nodes:
        x, y = pos_[node]
        coords.append((x, y))
    nodes_py = [
        [new_name, pos[0], pos[1]]
        for name, pos, new_name in zip(nodes, coords, nodes_ind)
    ]
    ds_nodes = pd.DataFrame(nodes_py, columns=["name", "x", "y"])

    ds_edges_py = []
    for (n0, n1) in graph.edges:
        ds_edges_py.append([redo[n0], redo[n1]])

    ds_edges = pd.DataFrame(ds_edges_py, columns=["source", "target"])

    hb = hammer_bundle(ds_nodes, ds_edges)
    hbnp = hb.to_numpy()
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0]
    start = 0

    segments = []
    for stop in splits:
        seg = hbnp[start:stop, :]
        segments.append(seg)
        start = stop
    df_geo = pd.DataFrame(columns=["lat", "lon", "text", "size", "color"])
    df_geo["lat"] = [i[1] for i in pos_.values()]
    df_geo["lon"] = [i[0] for i in pos_.values()]
    for name in graph.nodes:
        assert name in sirg_author_list
        #    print(name)
    df_geo["text"] = list(node for node in graph.nodes)

    fig = go.Figure()
    lats = []
    lons = []
    traces = []
    other_traces = []
    if streamlit:
        st.markdown(
            """Note only 1001 node edges are shown in interactive plot below, because making the full list of {0} edges interactive would take hours""".format(
                len(segments)
            )
        )
    if not just_nodes:
        for ind, seg in enumerate(tqdm(segments, title="Modifying Edges for Interactivity")):
            x0, y0 = seg[1, 0], seg[1, 1]  # graph.nodes[edge[0]]['pos']
            x1, y1 = seg[-1, 0], seg[-1, 1]  # graph.nodes[edge[1]]['pos']
            xx = seg[:, 0]
            yy = seg[:, 1]
            lats.append(xx)
            lons.append(yy)
            for i, j in enumerate(xx):
                if i > 0:
                    other_traces.append(
                        go.Scattergeo(
                            lon=[xx[i], xx[i - 1]],
                            lat=[yy[i], yy[i - 1]],
                            mode="lines",
                            showlegend=False,
                            hoverinfo='skip',
                            line=dict(width=0.5, color="blue"),
                        )
                    )
        fig.add_traces(other_traces)

    #with open('expensive_plotly_traces.p','wb') as f:
    #    pickle.dump(other_traces,f)
    fig.add_trace(
        go.Scattergeo(
            lat=df_geo["lat"],
            lon=df_geo["lon"],
            marker=dict(
                size=3,  # data['Confirmed-ref'],
                color=colors,
                opacity=1,
            ),
            text=list(graph.nodes),
            hovertemplate="%{text} <extra></extra>",
        )
    )
    # layout = fig["layout"]
    if streamlit:
        fig["layout"]["width"] = 1825
        fig["layout"]["height"] = 1825
        st.write(fig)
    return fig,colors


def main_plot_routine(both_sets_locations, missing_person_name, node_location_name):
    if os.path.exists("net_cache.p"):
        with open("net_cache.p", "rb") as f:
            temp = pickle.load(f)
        (
            G,
            second,
            lat,
            long,
            node_location_name,
            sirg_author_list,
            subnets,
            megnet,
        ) = temp
    else:
        subnets, megnet = recalculate_sirg_sub_nets()
        (
            G,
            second,
            lat,
            long,
            node_location_name,
            sirg_author_list,
        ) = remove_missing_persons_from_big_net(
            both_sets_locations, missing_person_name
        )
        temp = [
            G,
            second,
            lat,
            long,
            node_location_name,
            sirg_author_list,
            subnets,
            megnet,
        ]
        with open("net_cache.p", "wb") as f:
            pickle.dump(temp, f)
#second
    for name in sirg_author_list:
        second.add_node(name, pos=(second.nodes['Romain A. Dahan']['pos'][0], second.nodes['Romain A. Dahan']['pos'][1]))

    tab10 = sns.color_palette("bright")
    subnets = OrderedDict({k: v["g"] for k, v in subnets.items() if hasattr(v, "keys")})
    color_map = {}
    color_value_index = 0
    for k, v in subnets.items():
        color_value_index += 1
        color_map[k] = color_value_index

    sub_net_numb = {}
    for sbn, (k, v) in enumerate(subnets.items()):
        for sub_node in v.nodes:
            sub_net_numb[sub_node] = sbn
    colors = []
    for node in second.nodes:
        colors.append(tab10[sub_net_numb[node]])
    edge_colors = []
    for edge in second.edges:
        edge_colors.append(tab10[sub_net_numb[edge[0]]])
    df = pd.DataFrame({"Latitude": lat, "Longitude": long, "name": node_location_name})
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)
    )
    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

    graph_for_app(world,second,edge_colors,colors,sirg_author_list,tab10)

    fig, ax3, plt_bundled, segments = edge_bundle(
        second, world, colors, sirg_author_list, tab10
    )

    #fig, ax3, plt_bundled, segments = edge_bundle(
    #    second, world, colors, sirg_author_list, tab10
    #)
    assert segments is not None
    if plotly:
        st.markdown(""" Computing an interactive version of this map now.""")
    # In the meantime:
    # A lot of potential coauthors were excluded
    # see this list below:
    # Note sometimes this includes core SIRG authors, where initials are written. For contrast..
    # """
    # )
    missing_person_name = list(
        [k for k, v in both_sets_locations.items() if v[1] is None]
    )
    with open("missing_person_name.p", "rb") as f:
        missing_person_name = pickle.load(f)

    missing_person_name = [
        name for name in missing_person_name if name not in str("Brian Smith")
    ]
    ds_nodes = pd.DataFrame(missing_person_name, columns=["names"])
    # st.dataframe(ds_nodes)
    st.markdown("""Okay now making an interactive version of this plot ...""")
    assert segments is not None
    fig = edge_bundle_plotly(
        second, world, colors, sirg_author_list, tab10, segments=segments
    )

    # graph_for_app(pos,second)
    return plt, plt_bundled, ax3, fig
'''

from matplotlib.patches import Circle  # $matplotlib/patches.py
def circle( xy, radius, color="lightsteelblue", facecolor="none", alpha=1, ax=None ):
    """ add a circle to ax= or current axes
    """
        # from .../pylab_examples/ellipse_demo.py
    e = Circle( xy=xy, radius=radius )
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_edgecolor( color )
    e.set_facecolor( facecolor )  # "none" not None
    e.set_alpha( alpha )
    return ax
'''
def graph_for_app(world,second,edge_colors,colors,sirg_author_list,tab10):
    #core = nx.k_core(second, 98)
    #label_core = core.nodes  # nx.nodes(core,'labels')
    # fig = plt.figure()#frameon=False,figsize=(20,20))
    pos = nx.get_node_attributes(second, "pos")
    #fig, ax = plt.subplots(figsize=(40, 40))

    ax = world.plot(color="white", edgecolor="black", figsize=(60, 60))

    pos_all = nx.get_node_attributes(second, "pos")
    # ax1 = nx.draw(second,pos,node_size=21,  node_shape='o', alpha=0.7, edge_color='grey', width=0.1)
    ax2 = nx.draw_networkx_edges(
        second,
        pos_all,
        width=0.05,
        edge_color=edge_colors,
        style="solid",
        alpha=0.05,
        ax=ax,
        edge_vmin=None,
        edge_vmax=None,
        arrows=True,
        label=None,
    )  # , **kwds)

    ax3 = nx.draw_networkx_nodes(
        second,
        pos_all,
        node_size=21,
        node_color=colors,
        node_shape="o",
        alpha=1.0,
        vmin=None,
        vmax=None,
        linewidths=None,
        label=None,
    )  # , **kwds)
    ax6 = plt.scatter(-111.93316158417922,33.42152185, s=480, facecolors='r', edgecolors='r')
    plt.text(-111.93316158417922, 33.42152185,"Arizona State University",size=25)
    #bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
    #plt.text(-111.93316158417922, 33.42152185, "Arizona State University", ha="center", va="center", rotation=45,
    #            size=25,
    #            bbox=bbox_props)
    for i, v in enumerate(sirg_author_list):
        plt.scatter([], [], c=tab10[i], label="SIRG PI {}".format(v))
    plt.legend()
    st.pyplot(plt,use_column_width=False,width=None)
    #plt.savefig, bbox_inches="tight")

'''

def left_over():
    for node in G.nodes:
        print(node)
        try:
            lat_long = both_sets_locations[node]
            positions[node] = lat_long[1]
            G.nodes[node]["pos"] = lat_long[1]
            print(G.nodes[node]["pos"])
        except:
            pass

    pos = nx.get_node_attributes(second, "pos")
    ax = nx.draw(second, pos, node_size=4, figsize=(16, 12))
    df = pd.DataFrame({"Latitude": lat, "Longitude": long, "name": node_location_name})
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)
    )
    # st.write(gdf.head())
    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ax1 = world.plot(color="white", edgecolor="black", figsize=(16, 12))

    df = pd.DataFrame({"Latitude": lat, "Longitude": long, "name": node_location_name})
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)
    )
    # st.write(gdf.head())
    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ax = world.plot(color="white", edgecolor="black")  # ,figsize=(16,12))

    gdf.plot(ax=ax, color="red")
'''
