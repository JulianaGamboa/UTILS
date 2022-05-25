###4.3.6.
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt

A1= np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter = ",")
A2= np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter = ",")

G1=nx.to_networkx_graph(A1)
G2=nx.to_networkx_graph(A2)

def basic_net_stats (G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    degree_sequence = [d for n, d in G.degree()]
    print("Average degree: %.2f" % np.mean(degree_sequence))

basic_net_stats (G1) 
basic_net_stats (G2)     

def plot_degree_distribution(G):
    degree_sequence = [d for n, d in G.degree()]
    plt.hist(degree_sequence, histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree Distribution")

plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig("village_hist.pdf")
#Nótese que las distribuciones de los histogramas son diferentes a las de los modelos NetworkX.py (forma de campana)
#Al parecer los gráficos er no son los mejores modelos para social networks, de todos modos se los utiliza como modelos empíricos

G1.number_of_nodes()
nx.__version__

# nx.connected_component_subgraphs(G1) ### da error
# gen = nx.connected_component_subgraphs(G1)
# gen = G.subgraph(c).copy() for c in connected_components(G) #tb da error
# g = gen.__next__()
# type(g) 
