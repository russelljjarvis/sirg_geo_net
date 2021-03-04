import pickle
import os

with open("missing_person_name.p", "rb") as f:
    missing_person_name = pickle.load(f)
from rebuild_js import author_to_gaffiliations#, author_to_gh_index
from grab_js import coords_of_target_university, university_data_frame
from typing import List
def hIndex(citations:List)->int:
    """
    From:
    https://gist.github.com/restrepo/c5f8f9fd5504a3f93ae34dd10a5dd6b0
    https://github.com/kamyu104/LeetCode/blob/master/Python/h-index.py
    :type citations: List[int]
    :rtype: int

    # Given an array of citations (each citation is a non-negative integer)
    # of a researcher, write a function to compute the researcher's h-index.
    #
    # According to the definition of h-index on Wikipedia:
    # "A scientist has index h if h of his/her N papers have
    # at least h citations each, and the other N − h papers have
    # no more than h citations each."
    #
    # For example, given citations = [3, 0, 6, 1, 5],
    # which means the researcher has 5 papers in total
    # and each of them had received 3, 0, 6, 1, 5 citations respectively.
    # Since the researcher has 3 papers with at least 3 citations each and
    # the remaining two with no more than 3 citations each, their/her/his h-index is 3.
    #
    # Note: If there are several possible values for h, the maximum one is taken as the h-index.
    """
    print(sorted(citations))
    return sum(x >= i + 1 for i, x in enumerate(sorted(  list(citations), reverse=True)))


def author_to_gh_index(NAME):
    from scholarly import scholarly, ProxyGenerator

    pg = ProxyGenerator()
    pg.Tor_External(tor_sock_port=9050, tor_control_port=9051, tor_password="0r4ng3s")
    scholarly.use_proxy(pg)

    search_query = list(scholarly.search_author(NAME))
    citedby = []
    if len(search_query):
        for f in search_query[0]['filled']:
            print(f)

        for sq in search_query:
            if "citedby" in sq.keys():
                citedby.append(sq["citedby"])
            #print(sq.keys())
    hind = hIndex(citedby)
    return citedby,hind
try:
    with open("gad.p", "rb") as f:
        gad = pickle.load(f)

except:

    from scholarly import scholarly, ProxyGenerator

    pg = ProxyGenerator()
    pg.Tor_External(tor_sock_port=9050, tor_control_port=9051, tor_password="0r4ng3s")
    scholarly.use_proxy(pg)
    print("proxy enabled")

    ga = []
    gad = {}
    for name in missing_person_name:
        gad[name] = author_to_gaffiliations(name)
        ga.append(author_to_gaffiliations(name))

    with open("ga.p", "wb") as f:
        pickle.dump(ga, f)
    with open("gad.p", "wb") as f:
        pickle.dump(gad, f)
    with open("mega_net.p", "rb") as f:
        mg = pickle.load(f)

    from scholarly import scholarly, ProxyGenerator

    pg = ProxyGenerator()
    pg.Tor_External(tor_sock_port=9050, tor_control_port=9051, tor_password="0r4ng3s")
    scholarly.use_proxy(pg)
    print("proxy enabled")

    for name in mg.nodes:
        gad[name] = author_to_gaffiliations(name)
        with open("update_googl_big_job.p", "wb") as f:
            pickle.dump(gad, f)
def get_stuff():
    with open("mega_net.p", "rb") as f:
        mg = pickle.load(f)
    g_locations = {}
    citedbyd = {}

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
    #for name, institution in gad.items():
    for name in sirg_author_list:#mg.nodes:
        citedby,hi = author_to_gh_index(name)
        citedbyd[name] = hi
        print('hi {0}'.format(hi))
        with open("sirg_hindex_g_locations.p", "wb") as f:
            pickle.dump(citedbyd, f)
    sirg = pickle.load(open("sirg_hindex_g_locations.p","rb"))
    print(sirg)
            
'''
if os.path.exists("big_g_locations.p"):
    try:
        with open("big_g_locations.p", "rb") as f:
            g_locations = pickle.load(f)
    except:
        g_locations = {}
        for name, institution in gad.items():
            if institution is not None:
                # print(institution,'in google locations')
                search = institution.split(" ")
                if len(search) == 3:
                    xy = coords_of_target_university(
                        search[-3] + str(" ") + search[-2] + str(" ") + search[-1]
                    )
                if len(search) == 2:
                    xy = coords_of_target_university(search[-2] + str(" ") + search[-1])
                if len(search) == 1:
                    xy = coords_of_target_university(search[-1])
                if "xy" in locals():
                    check_none = xy[1]
                else:
                    check_none = None

                if check_none is not None:
                    g_locations[name] = xy
                else:
                    if "," in institution:
                        location_key = institution.split(",")[-1]
                        xy = coords_of_target_university(location_key)
                        #check_none = xy[1]
                        if xy is not None:
                            g_locations[name] = xy
                    else:
                        xy = coords_of_target_university(institution)
                        #check_none = xy[1]
                        if xy is not None:
                            g_locations[name] = xy

                if name in g_locations.keys():
                    print(name, g_locations[name])
                else:
                    print(name, "still not found")

    with open("big_g_locations.p", "wb") as f:
        pickle.dump(g_locations, f)
'''
