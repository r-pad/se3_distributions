import pygraphviz as pgh

G = pgh.AGraph()

G.add_node("A")
G.add_edge("A", "B")
G.draw("test.dot", prog="dot")
G.draw("test.png", prog="dot")
