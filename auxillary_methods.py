import semanticscholar as sch
import pprint
pprint = pprint.pprint

import matplotlib
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Text
import requests
from tqdm import tqdm
import streamlit as st
import pandas as pd
import networkx as networkx

nx = networkx
import pickle
import numpy as np
import plotly.graph_objects as go
import pandas as pd
#from datashader.bundling import hammer_bundle
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
from typing import List
'''
def unpaywall_semantic_links(NAME, tns):
    """
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    """
    dois, coauthors, titles = author_to_urls(NAME)
    visit_more_urls = []
    for index, doi_ in enumerate(tqdm(dois, title="Building Suitable Links")):
        r0 = str("https://api.semanticscholar.org/") + str(doi_)
        visit_more_urls.append(r0)
    return visit_more_urls
'''
'''
def visit_link(NAME):
    """
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    """
    more_links = {}
    author_results = []
    dois, coauthors, titles = author_to_urls(NAME)
    visit_urls.extend(more_links)
    for index, link in enumerate(
        tqdm(visit_urls, title="Text mining via API calls. Please wait.")
    ):
        requests.get(link)
    return author_results, visit_urls

'''
def draw_wstate_tree(G):

    # from networkx.drawing.nx_agraph import write_dot, graphviz_layout
    pos = nx.spring_layout(G)
    # pos = graphviz_layout(G, prog='dot')
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    matplotlib.use("Agg")

    plt.savefig("whole_net.png")
    st.write(plt.show())


# Custom function to create an edge between node x and node y, with a given text and width
def make_edge(x, y, text, width):
    return go.Scatter(
        x=x,
        y=y,
        line=dict(width=width, color="cornflowerblue"),
        hoverinfo="text",
        text=([text]),
        mode="lines",
    )


def plotly_sized(g):
    """
    https://towardsdatascience.com/tutorial-network-visualization-basics-with-networkx-and-plotly-and-a-little-nlp-57c9bbb55bb9
    """
    pos_ = nx.fruchterman_reingold_layout(g)
    #    x, y = pos_[node]

    # pos_ = nx.spring_layout(g)
    # For each edge, make an edge_trace, append to list
    edge_trace = []
    for edge in g.edges():
        weight = 1 + g.edges()[edge]["weight"]
        weight = 2.5 * np.log(weight)
        # if weight => 5:

        # print(weight)
        if g.edges()[edge]["weight"] > 0:
            char_1 = edge[0]
            char_2 = edge[1]
            x0, y0 = pos_[char_1]
            x1, y1 = pos_[char_2]
            text = char_1 + "--" + char_2 + ": " + str(g.edges()[edge]["weight"])
            trace = make_edge(
                [x0, x1, text],
                [y0, y1, text],
                text,
                width=weight,
            )
            edge_trace.append(trace)

    # Make a node trace
    # for node in g.nodes():

    # labels = [str(node) for node in g.nodes()]
    node_trace = go.Scatter(
        x=[],
        y=[],
        hoverinfo="none",
        text=([text]),
        mode="markers+text",
        marker=dict(color=[], size=[], line=None),
    )
    # marker=dict(symbol='circle-dot',
    #                            size=5,
    #                            color='#6959CD',
    #                            line=dict(color='rgb(50,50,50)', width=0.5))

    # For each node in g, get the position and size and add to the node_trace
    for node in g.nodes():
        x, y = pos_[node]
        # print(x,tuple(x),node_trace["x"])
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])
        # node_trace["marker"]["color"] += tuple(["cornflowerblue"])
        # node_trace['marker']['size'] += tuple([5*g.nodes()[node]['size']])
        node_trace["marker"]["size"] += tuple([0.45 * g.nodes()[node]["size"]])
        node_trace["text"] += tuple(["<b>" + str(node) + "</b>"])
    # Customize layout
    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",  # transparent background
        plot_bgcolor="rgba(0,0,0,0)",  # transparent 2nd background
        xaxis={"showgrid": False, "zeroline": False},  # no gridlines
        yaxis={"showgrid": False, "zeroline": False},  # no gridlines
    )  # Create figure
    layout["width"] = 725
    layout["height"] = 725

    fig = go.Figure(layout=layout)  # Add all edge traces
    for trace in edge_trace:
        fig.add_trace(trace)  # Add node trace
    fig.add_trace(node_trace)  # Remove legend
    fig.update_layout(showlegend=False)  # Remove tick labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)  # Show figure
    return fig
    # fig.show()


"""
def try_again(g):
    from holoviews.operation.datashader import datashade, bundle_graph
    import holoviews as hv

    #edges_df = g.edges#pd.read_csv('../assets/fb_edges.csv')
    ds_edges_py = [
        [n0, n1] for (n0, n1) in g.edges
    ]
    #edges_df = pd.DataFrame(ds_edges_py, columns=["source", "target"])
    fb_nodes = hv.Nodes(g.nodes)#.sort()
    fb_graph = hv.Graph((g.edges, fb_nodes), label='Entire Sirg Network')
    return fb_graph
    #

def ego_graph(g):
    # https://hvplot.holoviz.org/user_guide/NetworkX.html
    #import holoviews.networkx as hvnx
    from operator import itemgetter
    import holoviews as hv
    # Create a BA model graph
    #n = 1000
    #m = 2
    #G = nx.generators.barabasi_albert_graph(n, m)
    # find node with largest degree
    node_and_degree = g.degree()
    (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
    # Create ego graph of main hub
    hub_ego = nx.ego_graph(g, largest_hub)
    # Draw graph
    pos = nx.spring_layout(hub_ego)
    g = nx.draw(hub_ego, pos, node_color='blue', node_size=50, with_labels=False)
    # Draw ego as large and red
    #gnodes = nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], node_size=300, node_color='red')
    #result = g * gnodes
    st.write(hv.render(result, backend="bokeh"))
    return result

def data_shade(graph):
    # from sklearn.decomposition import PCA
    nodes = list(graph.nodes())
    weights = (
        np.asarray(list(map(lambda x: x[-1]["weight"], graph.edges(data=True)))) ** 2
    )
    pos_ = nx.get_attribute(pos)
    coords = []
    for node in graph.nodes:
        x, y = pos_[node]
        coords.append((x, y))
    nodes_py = [[name, pos[0], pos[1]] for name, pos in zip(nodes, coords)]
    ds_nodes = pd.DataFrame(nodes_py, columns=["name", "x", "y"])

    ds_edges_py = [
        [n0, n1] for (n0, n1) in graph.edges
    ]


    ds_edges = pd.DataFrame(ds_edges_py, columns=["source", "target"])
    hb = hammer_bundle(ds_nodes, ds_edges)
    fig = hb.plot(x="x", y="y", figsize=(9, 9))
    return fig
"""


def university_data_frame():

    world_universities = pd.read_csv("world-universities.csv")
    world_universities.rename(
        columns={
            "AD": "country",
            "University of Andorra": "university",
            "http://www.uda.ad/": "wesbite",
        },
        inplace=True,
    )


class tqdm:
    def __init__(self, iterable, title=None):
        if title:
            st.write(title)
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)


def network(coauthors, MAIN_AUTHOR):

    node_strengths = {}
    cnt = 0
    titles = {}
    for title, mini_net in coauthors:
        for names in mini_net:
            key = names["name"]["first"] + str(" ") + names["name"]["last"]
            # titles[key] = title
            if key not in node_strengths.keys():
                node_strengths[key] = 1
            else:
                node_strengths[key] += 1
            cnt += 1
    g = networkx.DiGraph()
    # for key,value in node_strengths.items():
    #

    for title, mini_net in coauthors:
        for names in mini_net:
            key = names["name"]["first"] + str(" ") + names["name"]["last"]
            g.add_node(key, label=title, size=node_strengths[key])

    if cnt > 100:
        st.markdown(
            """ Detected large degree of collaborators/interconnectdness {0} building network will take time ... """.format(
                cnt
            )
        )

    for title, mini_net in tqdm(
        coauthors,
        title="Queried authors, now building network structure and rendering plots",
    ):
        # build small worlds
        # from projection
        for i, namesi in enumerate(mini_net):
            keyi = namesi["name"]["first"] + str(" ") + namesi["name"]["last"]
            # to projection
            for j, namesj in enumerate(mini_net):
                keyj = namesj["name"]["first"] + str(" ") + namesj["name"]["last"]
                if i != j:
                    g.add_edge(keyi, keyj, weight=node_strengths[keyi])
    return g

def resolve_aliases_and_papers(paper, NAME):
    if "authors" in paper.keys():
        for author_ in paper["authors"]:
            if NAME in author_:
                if "aliases" in author_.keys():
                    aliases = author_["aliases"]
    return aliases

def semantic_scholar_alias(NAME):
    """
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    """

    author_results = []
    aliases = None
    dois, coauthors, titles = author_to_urls(NAME)
    #alias_dict = {}
    inv_alias_dict = {}
    velocity = {}
    for d in dois:
        paper = sch.paper(d, timeout=32)
        if 'authors' in paper.keys():
            all_coauthors = paper['authors']
            for co_name in all_coauthors:
                key = co_name["name"]

                author = sch.author(co_name['authorId'], timeout=32)

                if "aliases" in author.keys():
                    aliases = author["aliases"]
                    for a in aliases:
                        inv_alias_dict[a] = key
                pprint(inv_alias_dict)
                if not key in inv_alias_dict.keys():
                    inv_alias_dict[key] = key
                if "citationVelocity" in author.keys():
                    velocity[key] = author['citationVelocity']
    inv_alias_dict = {v:k for k,v in inv_alias_dict.items()}
    return inv_alias_dict,velocity

def specific_sirg_network(coauthors, MAIN_AUTHOR, sirg_author_list):
    '''
    Ensures main sirg authors are captured.

    '''

    init_alias_dict = {}
    for name in sirg_author_list:
        try:
            with open(str(name)+'_alias_dict.p','rb') as f:
                [init_alias_dict,velocity] = pickle.load(f)
        except:
            inv_alias_dict,velocity = semantic_scholar_alias(name)
            init_alias_dict.update(inv_alias_dict)
            with open(str(name)+'_alias_dict.p','wb') as f:
                pickle.dump([init_alias_dict,velocity],f)
    node_strengths = {}
    cnt = 0
    titles = {}
    for title, mini_net in coauthors:
        for names in mini_net:
            #resolve_aliases_and_papers(paper, NAME)
            key = names["name"]["first"] + str(" ") + names["name"]["last"]
            if key not in node_strengths.keys():
                node_strengths[key] = 1
            else:
                node_strengths[key] += 1
            cnt += 1
    g = networkx.DiGraph()
    sirg_core = networkx.DiGraph()
    for aut_name in sirg_author_list:
        g.add_node(aut_name, label=aut_name)#, size=node_strengths[key])
    inspect_aut = {}

    # for key,value in node_strengths.items():
    #

    for title, mini_net in coauthors:
        for names in mini_net:
            key = names["name"]["first"] + str(" ") + names["name"]["last"]
            g.add_node(key, label=key, size=node_strengths[key])


    for title, mini_net in coauthors:

        # build small worlds
        # from projection
        for i, namesi in enumerate(mini_net):
            keyi = namesi["name"]["first"] + str(" ") + namesi["name"]["last"]
            # to projection
            if keyi in sirg_author_list and keyi not in MAIN_AUTHOR:
                print('within sirg collaboration', keyi,MAIN_AUTHOR)
                print(keyi)

            for j, namesj in enumerate(mini_net):
                keyj = namesj["name"]["first"] + str(" ") + namesj["name"]["last"]
                #if i != j:
                #if keyi in MAIN_AUTHOR:
                #    print(keyi)
                #if keyi == keyj:
                #    print(keyi,keyj)
                if keyj in sirg_author_list and keyj not in MAIN_AUTHOR:
                    print('within sirg collaboration', keyi,MAIN_AUTHOR)
                    print(keyi,MAIN_AUTHOR,keyj)

                if keyi!=keyj:
                    g.add_edge(keyi, keyj)
                    if keyi not in MAIN_AUTHOR:
                        inspect_aut[keyi] =  keyj
    #import pprint;pprint.pprint(inspect_aut)
    #import pdb
    #pdb.set_trace()
    return g,sirg_core,init_alias_dict


def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link  # .split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'


def author_to_coauthor_network(auth_name: str = "") -> networkx.DiGraph():
    response = requests.get("https://dissem.in/api/search/?authors=" + str(auth_name))
    author_papers = response.json()
    coauthors = []
    titles = []
    list_of_dicts = []

    if len(author_papers["papers"]) == 0:
        st.markdown(
            """## That query lead to zero papers. \n Retry either adding or ommitting middle initial."""
        )

        return None
    for p in author_papers["papers"]:
        coauthors_ = p["authors"]
        title = p["title"]
        titles.append(title)
        coauthors.append((title, coauthors_))
        if "pdf_url" in p.keys():
            temp = {"title": p["title"], "Web_Link": p["pdf_url"]}
        else:
            temp = {"title": p["title"], "Web_Link": p["records"][0]["splash_url"]}
        list_of_dicts.append(temp)
    df = pd.DataFrame(list_of_dicts)

    with open(str(auth_name) + "_df.p", "wb") as f:
        pickle.dump(df, f)

    g = network(coauthors, auth_name)
    return g, df


def author_to_sirg_only_network(auth_name: str = "",sirg_author_list: List = []) -> networkx.DiGraph():
    response = requests.get("https://dissem.in/api/search/?authors=" + str(auth_name))
    author_papers = response.json()
    coauthors = []
    titles = []
    list_of_dicts = []
    for p in author_papers["papers"]:
        coauthors_ = p["authors"]
        title = p["title"]
        titles.append(title)

        #coauthors.append(coauthors_)
        coauthors.append((title, coauthors_))

    g = specific_sirg_network(coauthors, auth_name,sirg_author_list)
    return g

def push_frame_to_screen(df_links):
    df_links.drop_duplicates(subset="Web_Link", inplace=True)
    df_links["Web_Link"] = df_links["Web_Link"].apply(make_clickable)
    df_links = df_links.to_html(escape=False)
    st.write(df_links, unsafe_allow_html=True)




def author_to_urls(NAME):
    response = requests.get("https://dissem.in/api/search/?authors=" + str(NAME))
    author_papers = response.json()
    visit_urls = []
    coauthors = []
    titles = []
    for p in author_papers["papers"]:
        coauthors_ = p["authors"]
        title = p["title"]
        titles.append(title)
        coauthors.append(coauthors_)
        if "pdf_url" in p.keys():
            visit_urls.append(p["pdf_url"])
        records = p["records"][0]
        if "splash_url" in records.keys():
            visit_urls.append(records["splash_url"])
        if "doi" in records.keys():
            visit_urls.append(records["doi"])

    visit_urls = [i for i in visit_urls if "FIGSHARE" not in i]
    visit_urls = [i for i in visit_urls if "figshare" not in i]
    visit_urls = [i for i in visit_urls if "doi" in i]
    dois = []

    for link in visit_urls:
        if "https://doi.org/" in link:
            li = link.split("https://doi.org/")
            dois.append(li[1])
        if "http://dx.doi.org" in link:
            li = link.split("http://dx.doi.org")
            dois.append(li[1])
    return dois, coauthors, titles


def response_paper_to_url(p: dict = {}) -> List:
    if "pdf_url" in p.keys():
        visit_urls.append(p["pdf_url"])
    records = p["records"][0]
    if "splash_url" in records.keys():
        visit_urls.append(records["splash_url"])
    if "doi" in records.keys():
        visit_urls.append(records["doi"])

    visit_urls = [i for i in visit_urls if "FIGSHARE" not in i]
    visit_urls = [i for i in visit_urls if "figshare" not in i]
    visit_urls = [i for i in visit_urls if "doi" in i]
    dois = []

    for link in visit_urls:
        if "https://doi.org/" in link:
            li = link.split("https://doi.org/")
            dois.append(li[1])
        if "http://dx.doi.org" in link:
            li = link.split("http://dx.doi.org")
            dois.append(li[1])
    return dois


def get_id(e):
    """
    Determining the publication id is tricky since it involves looking
    in the element for the various places a cluster id can show up.
    If it can't find one it will use the data-cid which should be
    usable since it will be a dead end anyway: Scholar doesn't know of
    anything that cites it.
    """
    for a in e.find(".gs_fl a"):
        if "Cited by" in a.text:
            return get_cluster_id(a.attrs["href"])
        elif "versions" in a.text:
            return get_cluster_id(a.attrs["href"])
    if "data-cid" in e.attrs.keys():
        return e.attrs["data-cid"]
    else:
        print(e.attrs)


def get_citations(url, depth=1, pages=1):
    """
    Given a page of citations it will return bibliographic information
    for the source, target of a citation.
    """
    if url in seen:
        return

    html = get_html(url)
    seen.add(url)

    # get the publication that these citations reference.
    # Note: this can be None when starting with generic search results
    a = html.find("#gs_res_ccl_top a", first=True)
    if a:
        to_pub = {
            "id": get_cluster_id(url),
            "title": a.text,
        }
    else:
        to_pub = None

    for e in html.find("#gs_res_ccl_mid .gs_r"):
        try:
            from_pub = get_metadata(e)
            yield from_pub, to_pub
        except:
            pass
        # depth first search if we need to go deeper
        if depth > 0 and from_pub["cited_by_url"]:
            yield from get_citations(
                from_pub["cited_by_url"], depth=depth - 1, pages=pages
            )

    # get the next page if that's what they wanted
    if pages > 1:
        for link in html.find("#gs_n a"):
            if link.text == "Next":
                yield from get_citations(
                    "https://scholar.google.com" + link.attrs["href"],
                    depth=depth,
                    pages=pages - 1,
                )


def get_metadata(e):
    """
    Fetch the citation metadata from a citation element on the page.
    """
    article_id = get_id(e)

    a = e.find(".gs_rt a", first=True)
    if a:
        url = a.attrs["href"]
        title = a.text
    else:
        url = None
        title = e.find(".gs_rt .gs_ctu", first=True).text

    authors = source = website = None
    # try:
    meta = e.find(".gs_a", first=True).text
    # except:
    #    print(e)
    meta_parts = [m.strip() for m in re.split(r"\W-\W", meta)]
    if len(meta_parts) == 3:
        authors, source, website = meta_parts
    elif len(meta_parts) == 2:
        authors, source = meta_parts

    if source and "," in source:
        year = source.split(",")[-1].strip()
    else:
        year = source

    cited_by = cited_by_url = None
    for a in e.find(".gs_fl a"):
        if "Cited by" in a.text:
            cited_by = a.search("Cited by {:d}")[0]
            cited_by_url = "https://scholar.google.com" + a.attrs["href"]

    return {
        "id": article_id,
        "url": url,
        "title": title,
        "authors": authors,
        "year": year,
        "cited_by": cited_by,
        "cited_by_url": cited_by_url,
    }


def get_html(url):
    """
    get_html uses selenium to drive a browser to fetch a URL, and return a
    requests_html.HTML object for it.

    If there is a captcha challenge it will alert the user and wait until
    it has been completed.
    """
    global driver

    time.sleep(random.randint(1, 5))
    driver.get(url)
    while True:
        try:
            recap = driver.find_element_by_css_selector("#gs_captcha_ccl,#recaptcha")

        except NoSuchElementException:

            try:
                html = driver.find_element_by_css_selector("#gs_top").get_attribute(
                    "innerHTML"
                )
                return requests_html.HTML(html=html)
            except NoSuchElementException:
                print("google has blocked this browser, reopening")
                driver.close()
                driver = webdriver.Chrome()
                return get_html(url)

        print("... it's CAPTCHA time!\a ...")
        time.sleep(5)


def remove_nones(d):
    new_d = {}
    for k, v in d.items():
        if v is not None:
            new_d[k] = v
    return new_d


def to_json(g):
    j = {"nodes": [], "links": []}
    for node_id, node_attrs in g.nodes(True):
        node_attrs["id"] = node_id
        j["nodes"].append(node_attrs)
    for source, target, attrs in g.edges(data=True):
        j["links"].append({"source": source, "target": target})
    return j
