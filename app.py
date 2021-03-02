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

from auxillary_methods import author_to_coauthor_network, network  # ,try_again
import holoviews as hv
from auxillary_methods import (
    push_frame_to_screen,
    plotly_sized,
)  # , data_shade, draw_wstate_tree
import chord2
import shelve


def streamlit_maps():
    data = pd.DataFrame(
        {
            "awesome cities": ["Chicago", "Minneapolis", "Louisville", "Topeka"],
            "latitude": [41.868171, 44.979840, 38.257972, 39.030575],
            "longitude": [-87.667458, -93.272474, -85.765187, -95.702548],
        }
    )
    st.map(data)


def user_manual_fix_missing(list_of_dicts):
    st.sidebar.title("Add new or replace author location in dataframe")
    name = st.sidebar.text_input("Enter Author Name")
    address = st.sidebar.text_input("Insitution Address")
    longitude = st.sidebar.text_input("longitude")
    latitude = st.sidebar.text_input("latitude")
    if st.button("Add row"):
        list_of_dicts.append(
            {
                "name": name,
                "address": address,
                "longitude": longitude,
                "latitude": latitude,
            }
        )
    st.write(pd.DataFrame(get_data()))
    st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)


def disable_logo(plot, element):
    plot.state.toolbar.logo = None


hv.extension("bokeh", logo=False)
hv.output(size=300)
hv.plotting.bokeh.ElementPlot.finalize_hooks.append(disable_logo)
# "Antonin Delpeuch",

import geopandas
import plotly.graph_objects as go
import pandas as pd
import geopandas
import streamlit as st
import numpy as np
import pickle

from netgeovis2 import (
    main_plot_routine,
    identify_find_missing,
    remove_missing_persons_from_big_net,
)

with open("both_sets_locations.p", "rb") as f:
    both_sets_locations = pickle.load(f)

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


def get_table_download_link_csv(df):
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="authors.csv" target="_blank">Download csv file</a>'
    return href


def big_plot_job():
    if os.path.exists("missing_person.p"):
        with open("missing_person.p", "rb") as f:
            temp = pickle.load(f)
        [
            mg,
            both_sets_locations,
            missing_person_name,
            missing_person_location,
            both_sets_locations_missing,
            sirg_author_list,
        ] = temp

    else:
        (
            mg,
            both_sets_locations,
            missing_person_name,
            missing_person_location,
            both_sets_locations_missing,
            sirg_author_list,
        ) = identify_find_missing()
        # except:
        # both_sets_locations,missing_person_name,missing_person_location,both_sets_locations_missing = identify_find_missing()
        temp = [
            mg,
            both_sets_locations,
            missing_person_name,
            missing_person_location,
            both_sets_locations_missing,
            sirg_author_list,
        ]
        with open("missing_person.p", "wb") as f:
            pickle.dump(temp, f)

    # both_sets_locations
    # both_sets_locations.keys()

    node_positions = list(both_sets_locations.values())
    long_lat = [np[1] for np in node_positions if np[1] is not None]
    lat = [coord[0] for coord in long_lat]
    long = [coord[1] for coord in long_lat]
    node_location_name = [np[0] for np in node_positions if np[1] is not None]

    node_person = list([k for k, v in both_sets_locations.items() if v[0] is not None])
    # if False:
    if os.path.exists("big_g_locations.p"):
        try:
            with open("big_g_locations.p", "rb") as f:
                g_locations = pickle.load(f)
            both_sets_locations.update(g_locations)
            missing_person_name = list(
                set(missing_person_name) - set(g_locations.keys())
            )

        except:
            pass
    plt_unbundled, plt_bundled, ax3 = main_plot_routine(
        both_sets_locations, missing_person_name, node_location_name
    )
    # main_plot_routine(both_sets_locations, missing_person_name, node_location_name)


from PIL import Image


def main():

    st.markdown("""--------------""")
    st.title(
        """Geo Geographic Maps for whole SIRG network are time intensive to compute."""
    )
    image = Image.open("bundled_graph_static.png")
    st.markdown(
        """Big image try scrolling down..."""
    )

    st.image(
        image,
        caption="a cached: Bundled Geographic Network map of greater SIRG network",
        use_column_width=False,
        width=None
    )

    st.markdown(
        """Recomputing graphs and making an interactive version in case data was revised. In the meantime we will populate the screen while you wait with other stuff while we re-build them..."""
    )

    #identify_find_missing()
    figure_size = 200
    hv.output(size=figure_size)
    with open("mega_net.p", "rb") as f:
        mg = pickle.load(f)

    graph = hv.Graph.from_networkx(mg, networkx.layout.fruchterman_reingold_layout)
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
    label = "Coauthorship Network for whole SIRG network: "
    st.markdown(
        "<h3 style='text-align: left; color: black;'>" + label + "</h3>",
        unsafe_allow_html=True,
    )
    st.write(hv.render(graph, backend="bokeh"))

    # st.markdown("""Geo Geographic Maps computing now, this will take time""")
    #st.markdown(
    #    "<h1 style='text-align: left; color: black;'>"
    #    + str(
    #        "Geographic Maps for whole sirg network computing now, this will take time"
    #    )
    #    + "</h1>",
    #    unsafe_allow_html=True,
    #)
    st.markdown("""geo plots computing...""")


    if os.path.exists("missing_person.p"):
        with open("missing_person.p", "rb") as f:
            temp = pickle.load(f)
        [
            mg,
            both_sets_locations,
            missing_person_name,
            missing_person_location,
            both_sets_locations_missing,
            sirg_author_list,
        ] = temp

        # list_of_dicts = [ list(v) for k,v in both_sets_locations_missing.items()]
        # df = pd.DataFrame(list_of_dicts)
        # st.dataframe(df)

    big_plot_job()

    # st.markdown(
    #    """[My other science information dashboard app](https://agile-reaches-20338.herokuapp.com/)"""
    # )
    # """


# [Source Code:](https://github.com/russelljjarvis/CoauthorNetVis)
# """


if __name__ == "__main__":
    main()
