import streamlit as st
import pickle
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
from netgeovis2 import remove_missing_persons_from_big_net, identify_find_missing
import networkx as nx

def get_data():
    with open("missing_person_name.p", "rb") as f:
        missing_person_name = pickle.load(f)
    with open("net_cache.p", "rb") as f:
        temp = pickle.load(f)

    ( mg,
    both_sets_locations,
    missing_person_name,
    missing_person_location,
    both_sets_locations_missing,
    sirg_author_list ) = identify_find_missing()
    G, second, lat, long, node_location_name, sirg_author_list = remove_missing_persons_from_big_net(both_sets_locations, missing_person_name)
    missing_from_viz = set(G.nodes)-set(both_sets_locations_missing.keys())

    mus = set(G.nodes) & set(both_sets_locations_missing.keys())

    sub_graph = nx.subgraph(G, mus)
    list_of_edges = []
    for src,tgt in G.edges():
        list_of_edges.append({'src':src,'tgt':tgt})
    df_edges = pd.DataFrame(list_of_edges)


    both_sets_locations_ = {}
    long = {}
    lat = {}
    for k,v in both_sets_locations.items():
        #both_sets_locations_[k] = list([v[0],v[1]])
        long[k] = v[0]
        lat[k] = v[1]
    #both_sets_locations = both_sets_locations_

    df = pd.DataFrame(both_sets_locations)#,long,lat])
    df.rename(index={0:'institution',1:'lat_long'},inplace=True)
    df = df.T
    df["longitude"] = [i[0] for i in df['lat_long']]
    df["latitude"] = [i[1] for i in df['lat_long']]
    del df["lat_long"]
    with open("net_cache2.p", "wb") as f:
        pickle.dump(df,f)
    return df,missing_from_viz,df_edges


def get_table_download_link_csv_nodes(df):
    import base64

    # csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    # b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download SIRG geo-coded network node locations as csv file</a>'
    return href


def get_table_download_link_csv_edges(df):
    import base64

    # csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    # b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download SIRG geo network connectivity as csv file</a>'
    return href

def main():
    #try:
    #with open("net_cache2.p", "rb") as f:
    #    df = pickle.load(f)
    #except:
    df,missing_from_viz,df_edges = get_data()
    #    try:
    #        with open("net_cache2.p", "rb") as f:
    #            df = pickle.load(f)
    #    except:
    #        pass
    user_input = False
    my_expander_dl = st.sidebar.beta_expander("Download Data")
    if my_expander_dl:
        st.sidebar.markdown(get_table_download_link_csv_nodes(df), unsafe_allow_html=True)
        st.sidebar.markdown(get_table_download_link_csv_edges(df_edges), unsafe_allow_html=True)


    my_expander_miss = st.sidebar.beta_expander("Researchers Who are missing as their details could not be resolved")
    clicked_missing = my_expander_miss.button('see missing')

    if clicked_missing:
        st.markdown("total of {0} missing authors from the SIRG network".format(len(missing_from_viz)))

        st.markdown(missing_from_viz)
    my_expander_selecting = st.sidebar.beta_expander("Update researchers institution/location by selecting")


    if my_expander_selecting:
        selection = []
        selection.append(False)
        selection.extend(list(df.index))
        user_input0 = my_expander_selecting.radio("select name as appears here ",selection)
    my_expander_keyin = st.sidebar.beta_expander("Update researchers institution location by typing (new name)")
    if my_expander_keyin:
        user_input1 = my_expander_keyin.text_input("Enter name as appears here ie: 'Chelsea N. Cook'",False)
    #st.text(user_input)
    if user_input0 or user_input1!="False":
        if user_input0:
            user_input = user_input0
        else:
            user_input = user_input1

        #if user_input!="False":
        if user_input in df.columns:
            df = df.T
            st.markdown("You selected:")
            st.markdown(user_input)
            #if user_input in df.columns:
            st.write(df[user_input])
            user_input_inst = st.text_input("Enter University Update", str("Undefined"))
            df[user_input]["institution"] = user_input_inst
            user_input_loc_lat = st.text_input("Enter latitude Update", 0.0)
            df[user_input]["latitude"] = user_input_loc_lat
            user_input_loc_long = st.text_input("Enter longitude Update", 0.0)
            df[user_input]["longitude"] = user_input_loc_long
            st.markdown("Updated field:")
            df = df.T
            st.write(df.loc[user_input,:])
        else:
            df = df.T
            st.markdown("new entry:")
            df[user_input] = None
            user_input_inst = st.text_input("Enter University Update", str("Undefined"))
            df[user_input]["institution"] = user_input_inst
            user_input_loc_lat = st.text_input("Enter latitude Update", 0.0)
            df[user_input]["latitude"] = user_input_loc_lat
            user_input_loc_long = st.text_input("Enter longitude Update", 0.0)
            df[user_input]["longitude"] = user_input_loc_long
            st.markdown("Updated field:")
            df = df.T
            st.write(df.loc[user_input,:])

    else:
        selection = ['interactive','static']
        my_expander_plot_selecting = st.sidebar.beta_expander("Interogate interactive geo plot data?")
        user_input3 = my_expander_plot_selecting.radio("Interactive or static plot? ",selection)
        if user_input3=="static":
            dfw = pd.DataFrame({"Latitude": df["latitude"], "Longitude": df["longitude"], "name": df.index})
            gdf = geopandas.GeoDataFrame(
                dfw, geometry=geopandas.points_from_xy(dfw.Longitude, dfw.Latitude)
            )
            world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
            ax = world.plot(color="white", edgecolor="black", figsize=(60, 60))
            try:
                for x,y,name in zip(df["latitude"],df["longitude"],df.index):
                        ax0 = plt.scatter(x,y, s=280, facecolors='b', edgecolors='b')
            except:
                pass
            ax1 = plt.scatter(-111.93316158417922,33.42152185, s=680, facecolors='r', edgecolors='r')
            plt.text(-111.93316158417922, 33.42152185,"Arizona State University",size=25)
            st.pyplot(plt,use_column_width=False,width=None)
            #st.write(df)
        if user_input3=="interactive":
            df2 = pd.DataFrame(columns=["lat", "lon", "text", "size", "color"])
            df2["lat"] = df["latitude"]
            df2["lon"] = df["longitude"]
            tab10 = sns.color_palette("bright")
            colors = []
            cnt=0
            for i,_ in enumerate(df.index):
                if cnt==len(tab10)-1:
                    cnt=0
                else:
                    cnt+=1
                colors.append(tab10[cnt])

            mouse_over=[i+str(" ")+j for i,j in zip(list(df2.index),list(df["institution"]))]
            figg = px.scatter_geo(df2)#, locations="iso_alpha")
            figg.add_trace(
                go.Scattergeo(
                    lat=df2["lon"],
                    lon=df2["lat"],
                    marker=dict(
                        size=3.5,  # data['Confirmed-ref'],
                        opacity=0.5,
                        color=colors,
                    ),
                    text=mouse_over,
                    hovertemplate=mouse_over,
                )
            )
            st.write(figg)
        # Customize layout
        #    layout = go.Layout(
        #        paper_bgcolor="rgba(0,0,0,0)",  # transparent background
        #        plot_bgcolor="rgba(0,0,0,0)",  # transparent 2nd background
        #        xaxis={"showgrid": False, "zeroline": False},  # no gridlines
        #        yaxis={"showgrid": False, "zeroline": False},  # no gridlines
        #    )  # Create figure
        #    layout["width"] = 925
        #    layout["height"] = 925

    selection = ['data_frame','table']
    my_expander_table_selecting = st.sidebar.beta_expander("scroll table or frame?")
    user_input3 = my_expander_table_selecting.radio("scroll table or frame?",selection)


    if user_input3=="table":
        st.markdown("Node locations")
        st.table(df)
        st.markdown("node connectivity (source, target)")
        st.table(df_edges)
    if user_input3=="data_frame":
        st.markdown("# Node locations")
        st.write(df)
        st.markdown("# node connectivity (source, target)")
        st.write(df_edges)

if __name__ == "__main__":

    main()
