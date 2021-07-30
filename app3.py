import streamlit as st
import pickle
import pandas as pd
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
