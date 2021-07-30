import streamlit as st
import pickle
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def get_table_download_link_csv(df):
    import base64

    # csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    # b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
    return href

from netgeovis2 import remove_missing_persons_from_big_net, identify_find_missing

def main():
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

    user_input = False

    st.sidebar.markdown(get_table_download_link_csv(df), unsafe_allow_html=True)
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
            user_input_inst = st.text_input("Enter University Update", str("None"))
            df[user_input]["institution"] = user_input_inst
            user_input_loc_lat = st.text_input("Enter latitude Update", "None")
            df[user_input]["latitude"] = user_input_loc_lat
            user_input_loc_long = st.text_input("Enter longitude Update", "None")
            df[user_input]["longitude"] = user_input_loc_long
            st.markdown("Updated field:")
            df = df.T
            st.write(df.loc[user_input,:])
        else:
            df = df.T
            st.markdown("new entry:")
            df[user_input] = None
            user_input_inst = st.text_input("Enter University Update", str("None"))
            df[user_input]["institution"] = user_input_inst
            user_input_loc_lat = st.text_input("Enter latitude Update", "None")
            df[user_input]["latitude"] = user_input_loc_lat
            user_input_loc_long = st.text_input("Enter longitude Update", "None")
            df[user_input]["longitude"] = user_input_loc_long
            st.markdown("Updated field:")
            df = df.T
            st.write(df.loc[user_input,:])

    else:

        #my_expander_selecting = st.sidebar.beta_expander("Plot data")

        #if my_expander_selecting:
        #    selection = []
        #    selection.append(False)
        #    selection.extend(list(df.index))
        #    user_input0 = my_expander_selecting.radio("select name as appears here ",selection)
        #my_expander_keyin = st.sidebar.beta_expander("Update researchers institution location by typing (new name)")

        dfw = pd.DataFrame({"Latitude": df["latitude"], "Longitude": df["longitude"], "name": df.index})
        gdf = geopandas.GeoDataFrame(
            dfw, geometry=geopandas.points_from_xy(dfw.Longitude, dfw.Latitude)
        )
        world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
        ax = world.plot(color="white", edgecolor="black", figsize=(60, 60))
        for x,y,name in zip(df["latitude"],df["longitude"],df.index):
            ax0 = plt.scatter(x,y, s=280, facecolors='b', edgecolors='b')
        ax1 = plt.scatter(-111.93316158417922,33.42152185, s=680, facecolors='r', edgecolors='r')
        plt.text(-111.93316158417922, 33.42152185,"Arizona State University",size=25)
        st.pyplot(plt,use_column_width=False,width=None)
        fig = go.Figure()
        node_trace = go.Scatter(
            x=[],
            y=[],
            hoverinfo="none",
            text=([""]),
            mode="markers+text",
            marker=dict(color=[], size=[], line=None),
        )
        # marker=dict(symbol='circle-dot',
        #                            size=5,
        #                            color='#6959CD',
        #                            line=dict(color='rgb(50,50,50)', width=0.5))

        #for x,y,name,institution in zip(df["latitude"],df["longitude"],df.index,df["institution"]):
        #    node_trace["x"] += tuple([x])
        #    node_trace["y"] += tuple([y])
        #    node_trace["marker"]["color"] += tuple(["cornflowerblue"])
        #    node_trace["marker"]["size"] += tuple([5.45])# * g.nodes()[node]["size"]])

        df2 = pd.DataFrame(columns=["lat", "lon", "text", "size", "color"])
        df2["lat"] = df["latitude"]
        df2["lon"] = df["longitude"]
        import seaborn as sns
        tab10 = sns.color_palette("bright")
        #subnets = OrderedDict({k: v["g"] for k, v in subnets.items() if hasattr(v, "keys")})
        #color_map = {}
        #color_value_index = 0
        #for k, v in subnets.items():
        #    color_value_index += 1
        #    color_map[k] = color_value_index

        #sub_net_numb = {}
        #for sbn, (k, v) in enumerate(subnets.items()):
        #    for sub_node in v.nodes:
        #        sub_net_numb[sub_node] = sbn
        colors = []
        cnt=0
        for i,_ in enumerate(df.index):
            if cnt==len(tab10)-1:
                cnt=0
            else:
                cnt+=1
            colors.append(tab10[cnt])
        #figg = go.Figure(go.Scattergeo())
        #st.write(figg)

        #figg.update_geos(projection_type="natural earth")
        #fig.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
        import plotly.express as px

        figg = px.scatter_geo(df2)#, locations="iso_alpha")

        figg.add_trace(
            go.Scattergeo(
                lat=df2["lat"],
                lon=df2["lon"],
                marker=dict(
                    size=13,  # data['Confirmed-ref'],
                    opacity=0.5,
                    color=colors,
                ),
                text=list(df2.index),
                hovertemplate=list(df2.index),
            )
        )
        st.write(figg)

        # Customize layout
        layout = go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",  # transparent background
            plot_bgcolor="rgba(0,0,0,0)",  # transparent 2nd background
            xaxis={"showgrid": False, "zeroline": False},  # no gridlines
            yaxis={"showgrid": False, "zeroline": False},  # no gridlines
        )  # Create figure
        layout["width"] = 925
        layout["height"] = 925

        #fig = go.Figure(layout=layout)  # Add all edge traces


        #for trace in edge_trace:
        #    fig.add_trace(trace)  # Add node trace
        #fig.add_trace(node_trace)  # Remove legend

        #fig.add_traces(other_traces)


        # layout = fig["layout"]
        #fig["layout"]["width"] = 1825
        #fig["layout"]["height"] = 1825


        st.table(df)

        #st.write(dft.T)

    #except:
    #    my_expander.warning(
    #        "This user has not participated in survey, but people have answered questions about them"
    #    )
    #    my_expander.warning("Try toggling the transpose")
    #    my_expander.write(df[user_input])
    #    my_expander.write(df.loc[df.index.isin([user_input])])


    #import pdb
    #pdb.set_trace()


    #st.markdown("Processed anonymized network data that is visualized")

if __name__ == "__main__":

    main()
