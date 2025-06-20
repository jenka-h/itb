import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import community  # python-louvain package 

class ChromationarySystem:
    def __init__(self, n_candidates=512, similarity_threshold=10):
        self.n_candidates = n_candidates
        self.similarity_threshold = similarity_threshold
        self.candidate_colors = self.generate_candidate_colors()
        self.graph = self.build_similarity_graph()

    def generate_candidate_colors(self):
        colors = []
        step = int(np.cbrt(self.n_candidates))
        for r in np.linspace(0, 1, step):
            for g in np.linspace(0, 1, step):
                for b in np.linspace(0, 1, step):
                    colors.append((r, g, b))
        return colors

    def rgb_to_lab(self, rgb):
        rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=False)
        return convert_color(rgb_color, LabColor)

    def build_similarity_graph(self):
        Graph = nx.Graph()
        for i, c1 in enumerate(self.candidate_colors):
            Graph.add_node(c1)
            lab1 = self.rgb_to_lab(c1)
            for j in range(i + 1, len(self.candidate_colors)):
                c2 = self.candidate_colors[j]
                lab2 = self.rgb_to_lab(c2)
                delta_e = delta_e_cie2000(lab1, lab2)
                if delta_e < self.similarity_threshold: # There is a threshold for the color distance
                    Graph.add_edge(c1, c2, weight=1 / (1 + delta_e))  # Calculating delta E
        return Graph

    def recommend(self, input_colors, top_k=6):
        input_labs = [self.rgb_to_lab(rgb) for rgb in input_colors]

        # Cluster candidate colors using Louvain's algorithm
        partition = community.best_partition(self.graph)
        clusters = {}

        for color, cluster_id in partition.items():
            clusters.setdefault(cluster_id, []).append(color)

        # Select best-matching color from each cluster 
        scored_colors = []
        for cluster_colors in clusters.values():
            cluster_scores = []
            for c in cluster_colors:
                c_lab = self.rgb_to_lab(c)
                avg_dist = np.mean([delta_e_cie2000(c_lab, user_lab) for user_lab in input_labs])
                cluster_scores.append((avg_dist, c))
            cluster_scores.sort(key=lambda x: x[0])
            best_color = cluster_scores[0][1]
            scored_colors.append((cluster_scores[0][0], best_color))

        # Sort top-k across clusters by closeness
        scored_colors.sort(key=lambda x: x[0])
        return [color for _, color in scored_colors[:top_k]]

    def display_colors(self, colors, title="Recommended Palette"):
        plt.figure(figsize=(len(colors), 2))
        for i, rgb in enumerate(colors):
            plt.subplot(1, len(colors), i + 1)
            plt.imshow([[rgb]])
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Chromationary's Simulation
if __name__ == "__main__":
    system = ChromationarySystem()

    print("Enter your color(s) as RGB (0–255) values. Example: 255 0 0")
    print("Type 'finish' to stop entering colors.")

    user_colors = []

    while True:
        line = input("Enter a color (or type 'finish' to stop): ")
        if line.strip().lower() == "finish":
            break
        try:
            r, g, b = map(int, line.strip().split())
            user_colors.append((r / 255, g / 255, b / 255))
        except ValueError:
            print("Invalid format. Enter three numbers separated by spaces.")

    if not user_colors:
        print("No colors entered.")
    else:
        recommendations = system.recommend(user_colors, top_k=6)
        system.display_colors(user_colors, title="Your Input Colors")
        system.display_colors(recommendations, title="Recommended Colors")
