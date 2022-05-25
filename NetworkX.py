
import networkx as nx 

G = nx.Graph()
G.add_node(1)
G.add_nodes_from ([2,3])
G.add_nodes_from(["v", "u"])

G.nodes()
G.add_edge(1,2)
G.add_edge("u","v")
G.add_edges_from([(1,3), (1,4), (1,5), (1,6)]) #aunque no estén los nodos se pueden agregar links (y Py agrega los edges ;)

G.add_edge("u","w")
G.edges()
G.remove_node(2)
G.nodes()
G.remove_nodes_from([4,5])
G.nodes()
G.remove_edge(1,3)
G.edges()
G.remove_edges_from([(1,2), ("u","v")])
G.edges()
G.number_of_nodes()
G.number_of_edges()

import matplotlib.pyplot as plt
G = nx.karate_club_graph()
nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
plt.savefig("karate_graph.pdf")

G.degree() #se guarda en un diccionario donde los valores son los degrees, the key es the node ID
G.degree() [25] #accedo al nodo que quiero
G.degree(25) #obtengo lo mismo

### RANDOM GRAPHS ###
#####################

from scipy.stats import bernoulli
bernoulli.rvs(p=0.2) #tirar una moneda y que caiga True con una probabilidad de 0.2

def er_graph(N, p):
    """Generate an ER graph."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1< node2 and bernoulli.rvs(p=p) ==True: #node1<node2 se agrega para que no considere el nodo 1,2 y el 2,1; que sea sólo 1 combinación
                G.add_edge(node1, node2)
    return G

nx.draw(er_graph(50, 0.08), node_size=40, node_color="red")

plt.savefig("er1.pdf")

# G.number_of_nodes()
# G.number_of_edges()
    
# N=20
# p=0.2

def plot_degree_distribution(G):
    degree_sequence = [d for n, d in G.degree()]
    plt.hist(degree_sequence, histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree Distribution")

G1=er_graph(200, 0.08)
plot_degree_distribution(G1)
G2=er_graph(200, 0.08)
plot_degree_distribution(G2)
G3=er_graph(200, 0.08)
plot_degree_distribution(G3)
plt.savefig("hist.pdf")

#vemos que los 3 gráficos superpustos son muy similares 
























