import networkx as nx
G=nx.Graph()
G.add_edge(1,2,weight=7,color='red')
G.add_edge('jack',3,weight=8,color='red')
# for line in nx.generate_adjlist(G):
#     print(line)
# nx.write_gml(G, "test.gml")
print(G[1][3]['weight'])
