import pickle
import matplotlib

matplotlib.use("Agg")
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_agraph import write_dot, graphviz_layout


def draw_wstate_tree(G):
    # fig= plt.figure(figsize=(5,5))

    # pos = nx.spring_layout(G)
    pos = graphviz_layout(G, prog="dot")
    edge_labels = nx.get_edge_attributes(G, "label")
    # nx.draw(G, pos)nh
    # nx.draw_networkx_labels(G, pos, font_size=4)

    # thpos=nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=3)

    nx.draw_networkx_nodes(
        G, pos, node_shape="o", node_color="0.75"
    )  # ,node_size=1200,node_shape='o',node_color='0.75')

    nx.draw_networkx_edges(G, pos, width=0.7, edge_color="b")

    plt.axis("off")
    # plt.savefig("degree.png", bbox_inches="tight")

    plt.savefig("whole_net_not_dot_huge.png")
    plt.show()
    # st.write(plt.show())


with open("mega_net.p", "rb") as f:
    mg = pickle.load(f)
draw_wstate_tree(mg)
