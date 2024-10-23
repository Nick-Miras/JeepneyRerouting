import graph_search_algorithm_compilation
import networkx as nx
import geopandas as gpd
import pandas as pd
import random
import heapq
import math

# Step 1: Import the graph from 'metro_manila_process_osm_data.ipynb'
def load_graph_from_notebook(notebook_path):
    import nbformat
    from nbconvert import PythonExporter

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    exporter = PythonExporter()
    code, _ = exporter.from_notebook_node(notebook_content)

    exec_globals = {}
    exec(code, exec_globals)

    return exec_globals.get("G")

# Load Metro Manila graph from 'metro_manila_process_osm_data.ipynb'
G = load_graph_from_notebook('metro_manila_process_osm_data.ipynb')

# Step 2: Import the precipitation data
def load_precipitation_data():
    return pd.read_csv(r'C:\Users\Jed Padro\DataspellProjects\JeepneyRerouting\nasa_data\precipitation_data\combined_precipitation_data.csv')

# Load precipitation data
precipitation_data = load_precipitation_data()

# Step 3: Simulate flooding conditions based on precipitation data
def simulate_flood_conditions(G, precipitation_data, threshold=50):
    """
    Randomly simulates flooding by removing edges from the graph based on precipitation data.
    Roads (edges) are considered flooded if precipitation exceeds a certain threshold.
    """
    flooded_edges = []

    for u, v, data in G.edges(data=True):
        if random.choice(precipitation_data['precipitation']) > threshold:
            flooded_edges.append((u, v))
            # Mark the edge as flooded by removing it
            G.remove_edge(u, v)

    return flooded_edges

# Step 4: Monte Carlo Simulation to choose the best algorithm
def monte_carlo_simulation(G, source, target, precipitation_data, iterations=1000, flood_threshold=50):
    """
    Monte Carlo simulation that selects an optimized algorithm for rerouting
    based on the number of flooded roads using a variety of algorithms.
    """
    # All algorithms included - here we reference the functions, not call them yet
    algorithms = {
        "Dijkstra": graph_search_algorithm_compilation.dijkstra_search,
        "Floyd-Warshall": graph_search_algorithm_compilation.floyd_warshall_search,
        "Bellman-Ford": graph_search_algorithm_compilation.bellman_ford_search,
        "Bidirectional Search": graph_search_algorithm_compilation.bidirectional_search,
        "Dynamic Shortest Path": graph_search_algorithm_compilation.dynamic_shortest_path,
        "D* Lite": graph_search_algorithm_compilation.d_star_lite,
        "A* Search": graph_search_algorithm_compilation.a_star_search,
        "Monte Carlo Tree Search": graph_search_algorithm_compilation.monte_carlo_tree_search,
        "Yen's K-Shortest Paths": graph_search_algorithm_compilation.yen_k_shortest_paths,
        "Ant Colony Optimization": graph_search_algorithm_compilation.ant_colony_optimization,
    }

    algorithm_selection_count = {name: 0 for name in algorithms.keys()}

    for _ in range(iterations):
        # Create a deep copy of the graph to avoid modifying the original graph permanently
        G_copy = G.copy()

        # Simulate flooding conditions
        flooded_edges = simulate_flood_conditions(G_copy, precipitation_data, flood_threshold)

        # Severe flooding: Choose dynamic algorithms or Monte Carlo Tree Search
        if len(flooded_edges) > len(G.edges()) * 0.2:
            selected_algorithm_name = random.choice(["D* Lite", "Monte Carlo Tree Search", "Dynamic Shortest Path"])

        # Moderate flooding: Use traditional pathfinding with recalculations (e.g., Dijkstra, Bellman-Ford)
        elif len(flooded_edges) > len(G.edges()) * 0.1:
            selected_algorithm_name = random.choice(["Dijkstra", "Bellman-Ford", "Ant Colony Optimization", "A* Search"])

        # Light flooding: Consider faster algorithms for small updates (e.g., Bidirectional Search, Floyd-Warshall)
        else:
            selected_algorithm_name = random.choice(["Bidirectional Search", "Floyd-Warshall", "Yen's K-Shortest Paths"])

        # Run the selected algorithm on the copied graph
        selected_algorithm = algorithms[selected_algorithm_name]
        try:
            # Execute the selected algorithm on the graph
            selected_algorithm(G_copy, source, target, weight='weight')
        except Exception as e:
            print(f"Error running {selected_algorithm_name}: {e}")
            continue  # If an error occurs, skip this iteration

        # Track how many times each algorithm is selected
        algorithm_selection_count[selected_algorithm_name] += 1

    return algorithm_selection_count

# Define source and target nodes for the graph search algorithms
source = 'start_node'  # Replace with an actual node ID from the graph
target = 'end_node'    # Replace with an actual node ID from the graph

# Run the Monte Carlo Simulation
simulation_results = monte_carlo_simulation(G, source, target, precipitation_data, iterations=1000)

# Display results
print("Algorithm selection results after Monte Carlo simulation:")
for algo, count in simulation_results.items():
    print(f"{algo}: {count} times selected")
