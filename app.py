"""
Author: [Russell Jarvis](https://github.com/russelljjarvis)
"""


import shelve
import streamlit as st
import os
import pandas as pd
import pickle
import streamlit as st
#from auxillary_methods import ego_graph
from holoviews import opts, dim
from collections import Iterable
import networkx

from auxillary_methods import author_to_coauthor_network, network#,try_again
import holoviews as hv
from auxillary_methods import push_frame_to_screen, plotly_sized#, data_shade, draw_wstate_tree
import chord2
import shelve


# from flask import render_template
# def map_func():
# 	return render_template('geo_coding.html',apikey=api_key,latitude=latitude,longitude=longitude)#map.html is my HTML file name


def disable_logo(plot, element):
    plot.state.toolbar.logo = None


hv.extension("bokeh", logo=False)
hv.output(size=300)
hv.plotting.bokeh.ElementPlot.finalize_hooks.append(disable_logo)

author_list = [
    "Antonin Delpeuch",
    "Brian H Smith",
    "Christian Rabeling",
    "Jon F Harrison",
    "Juergen Liebig",
    "Stephen C Pratt",
    "Jennifer H Fewell",
    "Arianne J Cease",
    "Gro V Amdam"
]


def main():

    MAIN_AUTHOR = "Antonin Delpeuch"
    #"Brian H Smith"

    # options = [150, 175, 200, 50, 75, 100, 125]
    # figure_size = st.sidebar.radio("Figure size (smaller-faster)", options)
    figure_size = 200
    hv.output(size=figure_size)

    st.title("Create a Coauthorship Network")
    author_name1 = st.text_input("Enter Author Name:")
    """
    Note: Search applies [dissmin](https://dissemin.readthedocs.io/en/latest/api.html) API backend
    """
    options = tuple(author_list)
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

    with shelve.open("fast_graphs_splash.p") as db:
        flag = author_name in db
        if flag:
            fig = db[author_name]["chord"]
            graph = db[author_name]["graph"]
            df = db[author_name]["df"]
            if "fig_pln" in db[author_name].keys():
                fig_pln = db[author_name]["fig_pln"]
            if "g" in db[author_name].keys():
                g = db[author_name]["g"]

            # if 'fig_pln' not in db.keys():
            # 	g, df = author_to_coauthor_network(author_name)
            # 	fig_pln = plotly_sized(g)
            # 	db[author_name]['fig_pln'] = fig_pln
            # 	edges_df = networkx.to_pandas_adjacency(g)
            # 	fig = chord2.make_filled_chord(edges_df)
            # 	db[author_name]['chord'] = fig

            # fig_shade = data_shade(g)
            # db[author_name] = {'chord':fig,'graph':graph,'df':df}

        if not flag:
            # @st.cache(suppress_st_warning=True)
            # def wrapper(author_name):
            # 	g, df = author_to_coauthor_network(author_name)
            # 	return g,df
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
                db.pop(k,None)
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
    st.markdown("""--------------""")
    st.markdown(
        "<h3 style='text-align: left; color: black;'>"
        + str("Here are some of the publications dissmin used to build these networks.")
        + "</h3>",
        unsafe_allow_html=True,
    )
    push_frame_to_screen(df)
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
