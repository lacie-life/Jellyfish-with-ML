import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import collections

# Data Collection
X = [ [0, 0, 0, 1],
      [0, 0, 0, 0],
      [1, 0 ,0, 1],
      [2, 1, 0, 1],
      [2, 2, 1, 1],
      [2, 2, 1, 0],
      [1, 2, 1, 0],
      [0, 1, 0, 1],
      [0, 2 ,1, 1],
      [2, 1, 1, 1],
      [0, 1, 1, 0],
      [1, 1, 0, 0],
      [1, 0, 1, 1],
      [2, 1, 0, 0]]

Y = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

data_feature_names = ['outlook', 'temp', 'humidity', 'windy']

# Training
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

# Visualize data
dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
