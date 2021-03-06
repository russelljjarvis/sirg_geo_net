import streamlit as st
st.set_page_config(layout="wide")
import pickle
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
from netgeovis2 import remove_missing_persons_from_big_net, identify_find_missing
import networkx as nx
import numpy as np
import copy

try:
    from streamlit import beta_expander
    st.expander = beta_expander
    st.sidebar.expander = beta_expander

except:
    from streamlit import expander
    st.expander = expander
    st.sidebar.expander = expander

def disable_logo(plot, element):
    plot.state.toolbar.logo = None

@st.cache
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
    everyone = set(list(G.nodes))
    resolved = set(list(both_sets_locations.keys()))
    missing_from_viz = everyone.difference(resolved)

    #st.text(both_sets_locations.keys())
    mus = set(G.nodes) & set(both_sets_locations.keys())
    both_sets_locations["Theodre P. Pavlic"] = copy.copy(both_sets_locations["Brian H. Smith"])
    sub_graph = nx.subgraph(G, mus)
    list_of_edges = []
    for src,tgt in sub_graph.edges():
        if src in both_sets_locations.keys() and tgt in both_sets_locations.keys():
            list_of_edges.append({'src':src,'tgt':tgt})
    df_edges = pd.DataFrame(list_of_edges)


    both_sets_locations_ = {}
    long = {}
    lat = {}
    for k,v in both_sets_locations.items():
        long[k] = v[0]
        lat[k] = v[1]

    df = pd.DataFrame(both_sets_locations)#,long,lat])
    df.rename(index={0:'institution',1:'lat_long'},inplace=True)
    df = df.T
    df["longitude"] = [i[0] for i in df['lat_long']]
    df["latitude"] = [i[1] for i in df['lat_long']]
    del df["lat_long"]
    lats = df[df['institution']=="ARIZONA STATE UNIVERSITY"]['latitude'].values[0]
    lons = df[df['institution']=="ARIZONA STATE UNIVERSITY"]['longitude'].values[0]
    indexc = df[df['institution']=="Arizona State University"].index#['longitude'].values[0]

    for ind in indexc:
        df.loc[ind,'longitude'] = lons#df[df['institution']=="ARIZONA STATE UNIVERSITY"]['longitude'].values[0]
        df.loc[ind,'latitude'] = lats#df[df['institution']=="ARIZONA STATE UNIVERSITY"]['latitude'].values[0]
        df.loc[ind,'institution'] = "Arizona State University"
    #with open("net_cache2.p", "wb") as f:
    #    pickle.dump(df,f)

    with open("for_elsevier_api.p", "wb") as f:
        pickle.dump([missing_from_viz,second,G,mg],f)


    return df,missing_from_viz,df_edges,sirg_author_list,second


@st.cache
def fast_interact_net(dfj,second):
    tab10 = sns.color_palette("bright")
    colors = []
    cnt=0
    for i,_ in enumerate(dfj.index):
        if cnt==len(tab10)-1:
            cnt=0
        else:
            cnt+=1
        colors.append(tab10[cnt])

    asu_edge_x = []
    asu_edge_y = []
    for edge in second.edges():
        if dfj.loc[edge[0],"institution"]=="Arizona State University":
            x0, y0 = second.nodes[edge[0]]['pos']
            x1, y1 = second.nodes[edge[1]]['pos']
            asu_edge_x.append(x0)
            asu_edge_x.append(x1)
            asu_edge_y.append(y0)
            asu_edge_y.append(y1)
            #st.text("hit")
    edge_trace_asu = go.Scattergeo(
        lon=asu_edge_x, lat=asu_edge_y,
        mode="lines",
        showlegend=False,
        hoverinfo='skip',
        line=dict(width=0.215, color="blue"),
        )

    edge_x = []
    edge_y = []
    for edge in second.edges():
        x0, y0 = second.nodes[edge[0]]['pos']
        x1, y1 = second.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    edge_trace = go.Scattergeo(
        lon=edge_x, lat=edge_y,
        mode="lines",
        showlegend=False,
        hoverinfo='skip',
        line=dict(width=0.025, color="green"),
        )

    df2 = pd.DataFrame(columns=["lat", "lon"])#, "text", "size", "color"])
    df2["lat"] = dfj["latitude"]
    df2["lon"] = dfj["longitude"]
    df2["institution"] = dfj["institution"]

    mouse_over=[i+str(" ")+j for i,j in zip(list(dfj.index),list(dfj["institution"]))]



    figg = px.scatter_geo(df2)#,center={'lon':-111.93316158417922,'lat':33.42152185})#, locations="iso_alpha")

    selection = ['everyone','asu_only']#,'indirect_only']
    my_expander_direct = st.sidebar.expander("Direct or indirect Connnections?")
    asu_only = my_expander_direct.radio("Interactive or static plot? ",selection)
    if not asu_only =='asu_only':# and not asu_only =='indirect_only':
        figg.add_traces(edge_trace)
        figg.add_traces(edge_trace_asu)

    if asu_only =='asu_only':# or asu_only=='everyone':

        figg.add_traces(edge_trace_asu)


    asu_trace = go.Scattergeo(
            lat=[33.42152185],
            lon=[-111.93316158417922],
            marker=dict(
                size=16.0,  # data['Confirmed-ref'],
                opacity=0.53,
                color='red',
            ),
            text="ASU",
            hovertemplate="ASU",
        )

    figg.add_traces(asu_trace)
    asu_mt = go.Scattergeo(
            lat=df2[df2['institution']=="Arizona State University"]["lon"].values,
            lon=df2[df2['institution']=="Arizona State University"]["lat"].values,
            marker=dict(
                size=12.0,  # data['Confirmed-ref'],
                opacity=0.5,
                color='red',
            ),
            text=mouse_over,
            hovertemplate=mouse_over,
        )
    figg.add_traces(asu_mt)
    #st.text(df2[df2['institution']=="Arizona State University"])
    node_trace = go.Scattergeo(
            lat=df2["lon"],
            lon=df2["lat"],
            marker=dict(
                size=5.0,  # data['Confirmed-ref'],
                opacity=0.9,
                color=[],
            ),
            text=mouse_over,
            hovertemplate=mouse_over,
        )

    figg.add_trace(node_trace)


    #figg.update_layout(projection_type = "orthographic")


    figg["layout"]["width"] = 1425
    figg["layout"]["height"] = 1425
    return figg


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
    df,missing_from_viz,df_edges,sirg_author_list,second = get_data()

    #    except:
    #        pass
    #user_input = False
    #my_expander_dl = st.sidebar.beta_expander("Download Data")
    #if my_expander_dl:
    st.sidebar.markdown(get_table_download_link_csv_nodes(df), unsafe_allow_html=True)

    st.sidebar.markdown(get_table_download_link_csv_edges(df_edges), unsafe_allow_html=True)

    my_expander_pi = st.sidebar.expander("Social Insect Research Group Principle Investigators")
    clicked_pi = my_expander_pi.button('see PIs')

    if clicked_pi:
        my_expander_pi.markdown("Social Insect Research Group Principle Investigators {0}".format(sirg_author_list))


    my_expander_miss = st.sidebar.beta_expander("Researchers Who are missing as their details could not be resolved")
    clicked_missing = my_expander_miss.button('see missing')

    if clicked_missing:
        st.markdown("A total of {0} authors are included in the SIRG network because their affiliations and geolocations could be found with GScholar+Orcid".format(len(df)))

        st.markdown("Total of {0} missing authors from the SIRG network because their affiliations and geolocations could NOT be found with GScholar+Orcid".format(len(missing_from_viz)))

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
    if user_input0 or user_input1!="False":
        if user_input0:
            user_input = user_input0
        else:
            user_input = user_input1

        if user_input in df.columns:
            df = df.T
            st.markdown("You selected:")
            st.markdown(user_input)
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
        ##
        # add jitter
        ##

        lats = df[df['institution']=="ARIZONA STATE UNIVERSITY"]['latitude'].values[0]
        lons = df[df['institution']=="ARIZONA STATE UNIVERSITY"]['longitude'].values[0]
        indexc = df[df['institution']=="Arizona State University"].index#['longitude'].values[0]

        indexc = df[df['institution']=="Arizona State University"].index#['longitude'].values[0]
        #sigma = 1.5
        from scipy.spatial.distance import pdist

        p = pdist(df[['longitude', 'latitude']])                 # Get (n ** 2 - n) / 2 distances
        n = len(df)
        i, j = np.triu_indices(n, 1)              # indices of the upper triangle
                                                  # of a distance matrix.  Coincides
                                                  # with calculations from pdist

        too_close = np.zeros(n, bool)             # Initialize a mask for what's close
        np.logical_or.at(too_close, i, p <= .5)   # logically accumulate if any thing
                                                  # is too close per row
                                                  # Note: this will not dupe by the
                                                  # nature of grabbing the upper triangle

        shape = (too_close.sum(), 2)
        rng = np.random.rand(*shape) * .95         # The jittering
        dfj = copy.copy(df)
        dfj.loc[too_close, ['longitude', 'latitude']] += rng


        selection = ['interactive','static']
        my_expander_plot_selecting = st.sidebar.beta_expander("Interactive or colored static plot?")
        user_input3 = my_expander_plot_selecting.radio("Interactive or static plot? ",selection)
        if user_input3=="static":
            dfw = pd.DataFrame({"Latitude": dfj["latitude"], "Longitude": dfj["longitude"], "name": dfj.index})
            gdf = geopandas.GeoDataFrame(
                dfw, geometry=geopandas.points_from_xy(dfw.Longitude, dfw.Latitude)
            )
            world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
            ax = world.plot(color="white", edgecolor="black", figsize=(60, 60))
            try:
                for x,y,name in zip(dfj["latitude"],dfj["longitude"],dfj.index):
                    ax0 = plt.scatter(x,y, s=280, facecolors='b', edgecolors='b')
            except:
                pass
            ax1 = plt.scatter(-111.93316158417922,33.42152185, s=680, facecolors='r', edgecolors='r')
            plt.text(-111.93316158417922, 33.42152185,"Arizona State University",size=25)
            st.pyplot(plt,use_column_width=False,width=None)
        if user_input3=="interactive":
            figg = fast_interact_net(dfj,second)
            st.plotly_chart(figg, use_container_width=True)

            #st.write(figg)



    selection = [False,True]
    local_net = st.sidebar.radio("Jittered ASU local Net",selection)

    if local_net==True:
        df2 = pd.DataFrame(columns=["lat", "lon"])#, "text", "size", "color"])
        df2["lat"] = dfj["latitude"]
        df2["lon"] = dfj["longitude"]
        df2["institution"] = dfj["institution"]

        dfjj = df2[df2['institution']=="Arizona State University"]
        mouse_over=[i+str(" ")+j for i,j in zip(list(dfjj.index),list(dfjj["institution"]))]

        figt = px.scatter_geo(df2)#,center={'lon':-111.93316158417922,'lat':33.42152185})#, locations="iso_alpha")

        asu_mtt = go.Scattergeo(
                lat=dfjj[dfjj['institution']=="Arizona State University"]["lon"].values,
                lon=dfjj[dfjj['institution']=="Arizona State University"]["lat"].values,
                marker=dict(
                    size=1.0,  # data['Confirmed-ref'],
                    opacity=0.5,
                    color='red',
                ),
                text=mouse_over,
                hovertemplate=mouse_over,
            )
        figt.add_traces(asu_mtt)
        st.write(figt)

    selection = ['data_frame','table']
    my_expander_table_selecting = st.sidebar.beta_expander("scroll table or frame?")
    user_input3 = my_expander_table_selecting.radio("scroll table or frame?",selection)



    if user_input3=="table":
        st.markdown("# Node locations")
        st.table(df)
        st.markdown("# Connectivity Between Nodes Format: (source, target)")
        st.table(df_edges)
    if user_input3=="data_frame":
        st.markdown("# Node locations")
        st.write(df)
        st.markdown("# Connectivity Format: (source, target)")
        st.write(df_edges)

if __name__ == "__main__":

    main()
