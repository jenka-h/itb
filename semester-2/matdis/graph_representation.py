import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

n_candidates = 15  # Number of colors
similarity_threshold = 25  # Threshold for color similarity

def generate_candidate_colors(n):
    np.random.seed(0)
    return [tuple(np.random.rand(3)) for _ in range(n)]

def rgb_to_lab(rgb):
    rgb_color = sRGBColor(*rgb, is_upscaled=False)
    return convert_color(rgb_color, LabColor)

def build_graph(colors, threshold):
    G = nx.Graph()
    for i, c1 in enumerate(colors):
        G.add_node(i, color=c1)
        lab1 = rgb_to_lab(c1)
        for j in range(i + 1, len(colors)):
            c2 = colors[j]
            lab2 = rgb_to_lab(c2)
            delta = delta_e_cie2000(lab1, lab2)
            if delta < threshold:
                G.add_edge(i, j, weight=1 / (1 + delta))
    return G


colors = generate_candidate_colors(n_candidates)
G = build_graph(colors, similarity_threshold)
pos = nx.spring_layout(G, seed=42)
node_colors = [colors[i] for i in G.nodes()]
edge_labels = nx.get_edge_attributes(G, 'weight')

plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()},
    font_size=8
)
plt.title("Weighted Color Similarity Graph (12–15 Colors)")
plt.axis("off")
plt.tight_layout()
plt.show()
