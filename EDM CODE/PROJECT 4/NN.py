import matplotlib.pyplot as plt
import networkx as nx

def draw_network(layers, colors, title):
    pos = {}
    x_coords = {layer: i for i, layer in enumerate(layers.keys())}

    for layer, nodes in layers.items():
        y_spacing = 1.0 / (len(nodes) + 1)
        for i, node in enumerate(nodes):
            pos[node] = (x_coords[layer], 1 - (i + 1) * y_spacing)

    # Build directed graph
    G = nx.DiGraph()
    layer_names = list(layers.keys())
    for i in range(len(layer_names) - 1):
        for u in layers[layer_names[i]]:
            for v in layers[layer_names[i + 1]]:
                G.add_edge(u, v)

    node_color_map = {}
    for layer, nodes in layers.items():
        for node in nodes:
            node_color_map[node] = colors[layer]

    plt.figure(figsize=(12, 7))


    nx.draw_networkx_nodes(
        G, pos,
        node_size=1800,
        node_color=[node_color_map[n] for n in G.nodes()]
    )


    nx.draw_networkx_labels(
        G, pos,
        labels={node: node for node in G.nodes()},
        font_size=10, font_weight="bold", font_color='black'
    )

    nx.draw_networkx_edges(
        G, pos,
        arrows=True, arrowstyle="-|>", arrowsize=20,
        width=1.5, connectionstyle="arc3,rad=0.0"
    )

    for layer, nodes in layers.items():
        x = x_coords[layer]
        y = 1 + 0.05
        if "Hidden Layer" in layer:
            title_text = f"{layer} ({len(nodes)})"
        else:
            title_text = layer

        plt.text(x, y, title_text,
                 fontsize=14, fontweight='bold',
                 ha='center', va='bottom',
                 color=colors[layer])

    plt.title(title, fontsize=18, fontweight="bold")
    plt.axis("off")
    plt.show()

# ===== COLOR SCHEME =====
colors = {
    "Input": "SteelBlue",
    "Hidden Layer 1": "mediumseagreen",
    "Hidden Layer 2": "gold",
    "Hidden Layer 3": "mediumpurple",
    "Output": '#E45756'
}

# === Surface Waviness  NETWORK===
layers_sw = {
    "Input": ["Pluse\nOn", "Pluse\nOff", "Servo\nVoltage", " Current","Angle"],
    "Hidden Layer 1": [f"H1_{i+1}" for i in range(6)],
    "Hidden Layer 2": [f"H2_{i+1}" for i in range(4)],
    "Hidden Layer 3": [f"H3_{i+1}" for i in range(2)],
    "Output": ["Surface\nWaviness"]
}

# === SVR Network ===
layers_mer = {
     "Input": ["Pluse\nOn", "Pluse\nOff", "Servo\nVoltage", " Current","Angle"],
    "Hidden Layer 1": [f"H1_{i+1}" for i in range(5)],
    "Hidden Layer 2": [f"H2_{i+1}" for i in range(3)],
    "Hidden Layer 3": [f"H3_{i+1}" for i in range(2)],
    "Output": ["MVR"]
}

# === Kerf Distance   Network ===
layers_kg = {
    "Input": ["Pluse\nOn", "Pluse\nOff", "Servo\nVoltage", " Current","Angle"],
    "Hidden Layer 1": [f"H1_{i+1}" for i in range(6)],
    "Hidden Layer 2": [f"H2_{i+1}" for i in range(4)],
    "Hidden Layer 3": [f"H3_{i+1}" for i in range(2)],
    "Output": ["Kerf\nClearence"]
}

# Draw all updated network diagrams
draw_network(layers_sw, colors, "Neural Network Architecture: Surface Waviness")
draw_network(layers_mer, colors, "Neural Network Architecture: Material Vaporization Rate ")
draw_network(layers_kg, colors, "Neural Network Architecture: Kerf Distance ")
