import random

import networkx as nx
import matplotlib.pyplot as plt

class GraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node):
        self.graph.add_node(node)

    def add_nodes(self, nodes):
        self.graph.add_nodes_from(nodes)

    def update_node(self, old_node, new_node):
        if old_node in self.graph:
            in_edges = [(u, v, d) for u, v, d in self.graph.in_edges(old_node, data=True)]
            out_edges = [(u, v, d) for u, v, d in self.graph.out_edges(old_node, data=True)]
            self.graph.remove_node(old_node)
            self.graph.add_node(new_node)
            for u, v, d in in_edges:
                self.graph.add_edge(u, new_node, weight=d['weight'])
            for u, v, d in out_edges:
                self.graph.add_edge(new_node, v, weight=d['weight'])

    def delete_node(self, node):
        self.graph.remove_node(node)

    def has_edge(self, node1, node2):
        return self.graph.has_edge(node1, node2)

    def add_edge(self, node1, node2, weight=1):
        self.graph.add_edge(node1, node2, weight=weight)

    def delete_edge(self, node1, node2):
        self.graph.remove_edge(node1, node2)

    def get_edge_weight(self, task_a, task_b):
        if self.graph.has_edge(task_a, task_b):
            return self.graph[task_a][task_b]['weight']
        else:
            return None

    def visualize(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold')
        labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        plt.show()

    def select_random_tasks(self, initial_node, final_node):
        nodes = list(self.graph.nodes)
        if len(nodes) < 2:
            raise ValueError("Graph must have at least 2 nodes to select random states.")

        if len(nodes) == 2:
            node1, node2 = nodes
        else:
            # Remove only the final node from the list
            nodes_filtered = [node for node in nodes if node != final_node]

            # Randomly select one node from the filtered list
            node1 = random.choice(nodes_filtered)

            # Add the final node back into the list
            nodes_filtered.append(final_node)

            # Remove the selected node1 and the initial node from the list
            nodes_filtered.remove(node1)
            if initial_node in nodes_filtered:
                nodes_filtered.remove(initial_node)

            # Randomly select the second node from the updated list
            node2 = random.choice(nodes_filtered)

        return node1, node2
            
    def get_neighbors(self, node):
        neighbors = []
        for neighbor, edge_data in self.graph[node].items():
            edge_cost = edge_data['weight']
            neighbors.append((neighbor, edge_cost))
        return neighbors