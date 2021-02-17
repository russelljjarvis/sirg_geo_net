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

def draw_wstate_tree(G):

    #from networkx.drawing.nx_agraph import write_dot, graphviz_layout
    pos = nx.spring_layout(G)
    #pos = graphviz_layout(G, prog='dot')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    nx.draw_networkx_labels(G, pos, font_size=10)
    matplotlib.use('Agg')

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
    #bundled = bundle_graph(fb_graph)
    #bundled
    print(fb_graph)
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
    pos_ = nx.fruchterman_reingold_layout(graph)
    coords = []
    for node in graph.nodes:
        x, y = pos_[node]
        coords.append((x, y))
    nodes_py = [[name, pos[0], pos[1]] for name, pos in zip(nodes, coords)]
    ds_nodes = pd.DataFrame(nodes_py, columns=["name", "x", "y"])
    #print(graph.edges)
    #import pdb
    #pdb.set_trace()
    #ds_edges_py = [[int(n0), int(n1)] for (n0, n1) in graph.edges]

    ds_edges_py = [
        [n0, n1] for (n0, n1) in graph.edges
    ]


    ds_edges = pd.DataFrame(ds_edges_py, columns=["source", "target"])
    hb = hammer_bundle(ds_nodes, ds_edges)
    fig = hb.plot(x="x", y="y", figsize=(9, 9))
    return fig


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


def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = link  # .split('=')[1]
    return f'<a target="_blank" href="{link}">{text}</a>'


def author_to_coauthor_network(name: str = "") -> networkx.DiGraph():
    response = requests.get("https://dissem.in/api/search/?authors=" + str(name))
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

    with open(str(name) + "_df.p", "wb") as f:
        pickle.dump(df, f)

    g = network(coauthors, name)
    # with open(str(name)+".p","wb") as f:
    # 	pickle.dump(g,f)
    return g, df


def push_frame_to_screen(df_links):
    df_links.drop_duplicates(subset="Web_Link", inplace=True)
    df_links["Web_Link"] = df_links["Web_Link"].apply(make_clickable)
    df_links = df_links.to_html(escape=False)
    st.write(df_links, unsafe_allow_html=True)


def get_cluster_id(url):
    """
    Google assign a cluster identifier to a group of web documents
    that appear to be the same publication in different places on the web.
    How they do this is a bit of a mystery, but this identifier is
    important since it uniquely identifies the publication.
    """
    vals = parse_qs(urlparse(url).query).get("cluster", [])
    if len(vals) == 1:
        return vals[0]
    else:
        vals = parse_qs(urlparse(url).query).get("cites", [])
        print(vals)
        if len(vals) == 1:
            return vals[0]
    return None


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
