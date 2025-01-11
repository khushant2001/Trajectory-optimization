"""
This is just a demo program which will be worked on later. Consider it to be a placeholder!
I am learning and want to try the Graph of Convex Sets approach!
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define convex sets as rectangular regions (x_min, x_max, y_min, y_max)
convex_sets = [
    {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 2},
    {"x_min": 1, "x_max": 3, "y_min": 0, "y_max": 2},
    {"x_min": 0, "x_max": 1, "y_min": 2, "y_max": 4},
    {"x_min": 1, "x_max": 3, "y_min": 2, "y_max": 4},
]

# Assign a unique color to each convex set
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']

# Create a graph representing the convex sets and transitions
G = nx.Graph()

# Add nodes to the graph (each node corresponds to a convex region)
for i in range(len(convex_sets)):
    G.add_node(i)

# Add edges representing transitions between adjacent convex sets
G.add_edge(0, 1)
G.add_edge(0, 2)
G.add_edge(1, 3)
G.add_edge(2, 3)

# Coordinates for node locations (center of each convex set)
node_positions = {
    0: [(convex_sets[0]['x_min'] + convex_sets[0]['x_max']) / 2,
        (convex_sets[0]['y_min'] + convex_sets[0]['y_max']) / 2],
    1: [(convex_sets[1]['x_min'] + convex_sets[1]['x_max']) / 2,
        (convex_sets[1]['y_min'] + convex_sets[1]['y_max']) / 2],
    2: [(convex_sets[2]['x_min'] + convex_sets[2]['x_max']) / 2,
        (convex_sets[2]['y_min'] + convex_sets[2]['y_max']) / 2],
    3: [(convex_sets[3]['x_min'] + convex_sets[3]['x_max']) / 2,
        (convex_sets[3]['y_min'] + convex_sets[3]['y_max']) / 2],
}

# Visualize the convex sets
plt.figure()

for i, convex_set in enumerate(convex_sets):
    plt.gca().add_patch(plt.Rectangle(
        (convex_set["x_min"], convex_set["y_min"]),
        convex_set["x_max"] - convex_set["x_min"],
        convex_set["y_max"] - convex_set["y_min"],
        fill=True, color=colors[i], alpha=0.5, label=f"Convex Set {i}"
    ))

# Draw transitions as edges
nx.draw_networkx_edges(G, pos=node_positions, ax=plt.gca(), edge_color='black', arrows=True)

# Plot the convex set centers and annotate them
for i in range(len(convex_sets)):
    plt.scatter(node_positions[i][0], node_positions[i][1], color='black', zorder=5)
    plt.text(node_positions[i][0], node_positions[i][1], f'{i}', color='black',
             fontsize=12, ha='center', va='center')

# Add a legend to differentiate sets
plt.xlim(-1, 4)
plt.ylim(-1, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title("Convex Sets and Transitions")
plt.show()