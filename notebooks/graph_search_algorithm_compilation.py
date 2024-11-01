from joblib import Parallel, delayed
from collections import deque
import networkx as nx
import random
import heapq
import math


# 1
def dijkstra_search(G, source, target, weight="weight"):
    """
    Dijkstra's Algorithm to find the shortest path between two nodes in a weighted graph.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute that holds the weight (default is "weight")

    Returns: list of nodes representing the shortest path from source to target
    """

    # Priority queue to store (distance, node, path)
    queue = [(0, source, [source])]
    # Dictionary to store the shortest known distance to each node
    distances = {source: 0}
    # Set to track visited nodes
    visited = set()

    while queue:
        # Get the node with the smallest distance
        (current_dist, current_node, path) = heapq.heappop(queue)

        # If the node is the target, return the path
        if current_node == target:
            return path

        # If the node has already been visited, skip it
        if current_node in visited:
            continue
        visited.add(current_node)

        # Explore neighbors of the current node
        for neighbor in G.neighbors(current_node):
            edge_weight = G[current_node][neighbor].get(weight, 1)  # Default weight is 1 if none is provided
            new_dist = current_dist + edge_weight

            # If a shorter path to the neighbor is found, update the queue and distances
            if new_dist < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor, path + [neighbor]))

    # If no path is found, return None
    return None


# 2
def floyd_warshall_search(G, source, target, weight="weight"):
    """
    Floyd-Warshall algorithm to find the shortest path between two nodes in a weighted graph.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute that holds the weight (default is "weight")

    Returns: list of nodes representing the shortest path from source to target
    """

    # Initialize the distance and predecessor dictionaries
    dist = {node: {node2: float('inf') for node2 in G.nodes()} for node in G.nodes()}
    next_node = {node: {node2: None for node2 in G.nodes()} for node in G.nodes()}

    # Set the distance for each edge in the graph
    for u, v, data in G.edges(data=True):
        dist[u][v] = data.get(weight, 1)  # Default weight is 1 if none is provided
        dist[v][u] = data.get(weight, 1)  # If undirected, set both directions
        next_node[u][v] = v
        next_node[v][u] = u

    # Set the distance from each node to itself to 0
    for node in G.nodes():
        dist[node][node] = 0

    # Floyd-Warshall main loop
    for k in G.nodes():
        for i in G.nodes():
            for j in G.nodes():
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    # Reconstruct the path from source to target using next_node
    if next_node[source][target] is None:
        return None  # No path exists

    path = []
    current_node = source
    while current_node != target:
        path.append(current_node)
        current_node = next_node[current_node][target]
        if current_node is None:
            return None  # No path exists

    path.append(target)
    return path


# 3
def bellman_ford_search(G, source, target, weight="weight"):
    """
    Bellman-Ford Algorithm to find the shortest path from source to target in a weighted graph.
    Handles negative weights and detects negative weight cycles.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute that holds the weight (default is "weight")

    Returns: list of nodes representing the shortest path from source to target or None if no path exists.
    Raises: ValueError if a negative weight cycle is detected.
    """

    # Step 1: Initialize distances and predecessors
    distances = {node: float('inf') for node in G.nodes()}
    predecessors = {node: None for node in G.nodes()}

    distances[source] = 0

    # Step 2: Relax edges |V| - 1 times (V is the number of nodes)
    for _ in range(len(G.nodes()) - 1):
        for u, v, data in G.edges(data=True):
            edge_weight = data.get(weight, 1)
            if distances[u] + edge_weight < distances[v]:
                distances[v] = distances[u] + edge_weight
                predecessors[v] = u
            if distances[v] + edge_weight < distances[u]:  # If undirected, relax the reverse direction too
                distances[u] = distances[v] + edge_weight
                predecessors[u] = v

    # Step 3: Check for negative weight cycles
    for u, v, data in G.edges(data=True):
        edge_weight = data.get(weight, 1)
        if distances[u] + edge_weight < distances[v]:
            raise ValueError("Graph contains a negative weight cycle")
        if distances[v] + edge_weight < distances[u]:  # Check reverse direction
            raise ValueError("Graph contains a negative weight cycle")

    # Step 4: Reconstruct the path from source to target
    if distances[target] == float('inf'):
        return None  # No path exists

    path = []
    current_node = target
    while current_node is not None:
        path.insert(0, current_node)
        current_node = predecessors[current_node]

    return path


# 4
def bidirectional_search(G, source, target, weight="weight"):
    """
    Bidirectional Search to find the shortest path from source to target in a weighted graph.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute name for weights (default is "weight").

    Returns: list of nodes representing the shortest path from source to target.
    """

    if source == target:
        return [source]

    # Priority queues for Dijkstra-like expansion from source and target
    forward_queue = [(0, [source])]
    backward_queue = [(0, [target])]

    # Dictionaries to track the shortest path distances from both directions
    forward_visited = {source: (0, [source])}
    backward_visited = {target: (0, [target])}

    while forward_queue and backward_queue:
        # Expand from the forward side
        if forward_queue:
            forward_dist, forward_path = heapq.heappop(forward_queue)
            current_node_forward = forward_path[-1]

            # Explore neighbors with cumulative weights
            for neighbor in G.neighbors(current_node_forward):
                edge_weight = G[current_node_forward][neighbor].get(weight, 1)
                new_dist = forward_dist + edge_weight

                if neighbor not in forward_visited or new_dist < forward_visited[neighbor][0]:
                    new_path = forward_path + [neighbor]
                    forward_visited[neighbor] = (new_dist, new_path)
                    heapq.heappush(forward_queue, (new_dist, new_path))

                    # Check if we meet the backward search
                    if neighbor in backward_visited:
                        return new_path[:-1] + backward_visited[neighbor][1][::-1]

        # Expand from the backward side
        if backward_queue:
            backward_dist, backward_path = heapq.heappop(backward_queue)
            current_node_backward = backward_path[-1]

            # Explore neighbors with cumulative weights
            for neighbor in G.neighbors(current_node_backward):
                edge_weight = G[current_node_backward][neighbor].get(weight, 1)
                new_dist = backward_dist + edge_weight

                if neighbor not in backward_visited or new_dist < backward_visited[neighbor][0]:
                    new_path = backward_path + [neighbor]
                    backward_visited[neighbor] = (new_dist, new_path)
                    heapq.heappush(backward_queue, (new_dist, new_path))

                    # Check if we meet the forward search
                    if neighbor in forward_visited:
                        return forward_visited[neighbor][1] + backward_visited[neighbor][1][::-1][1:]

    # If no path is found, return None
    return None



# 5
def dynamic_shortest_path(G, source, target, weight="weight", updated_edges=None):
    """
    Dynamic Shortest Path Algorithm in a weighted graph.
    Updates shortest path dynamically when edge weights are updated.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute that holds the weight (default is "weight")
    updated_edges: list of (u, v, new_weight) representing edges whose weights have changed.

    Returns: list of nodes representing the shortest path from source to target.
    """

    # Step 1: Initial shortest path computation using Dijkstra's algorithm
    def dijkstra(G, source, target):
        # Priority queue for Dijkstra's algorithm
        priority_queue = [(0, source)]
        shortest_path = {source: 0}
        predecessors = {source: None}

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node == target:
                break

            for neighbor, data in G[current_node].items():
                edge_weight = data.get(weight, 1)
                distance = current_distance + edge_weight

                if neighbor not in shortest_path or distance < shortest_path[neighbor]:
                    shortest_path[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        return shortest_path, predecessors

    # Reconstruct the path from predecessors
    def reconstruct_path(predecessors, target):
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors.get(current)
        return path[::-1]

    # Step 2: Update the graph with the modified edges
    if updated_edges:
        for u, v, new_weight in updated_edges:
            if G.has_edge(u, v):
                G[u][v][weight] = new_weight

    # Step 3: Recompute shortest path after graph modification
    shortest_path, predecessors = dijkstra(G, source, target)

    # Step 4: Return the shortest path from source to target
    if target in shortest_path:
        return reconstruct_path(predecessors, target)
    else:
        return None


# 6
def d_star_lite(G, source, target, weight="weight", heuristic=lambda u, v: 0):
    """
    D* Lite Algorithm for dynamic path planning in a weighted graph.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute that holds the weight (default is "weight")
    heuristic: a function that estimates the cost from a node to the target

    Returns: list of nodes representing the shortest path from source to target.
    """

    # Initialize variables
    open_set = []
    g_score = {node: float('inf') for node in G.nodes}
    rhs = {node: float('inf') for node in G.nodes}
    g_score[source] = 0
    rhs[source] = 0

    heapq.heappush(open_set, (0, source))

    def compute_key(node):
        return (min(g_score[node], rhs[node]) + heuristic(node, target), min(g_score[node], rhs[node]))

    def update_vertex(node):
        if node != source:
            rhs[node] = min(G[nbr][node].get(weight, float('inf')) + g_score[nbr] for nbr in G.neighbors(node))

        if g_score[node] != rhs[node]:
            heapq.heappush(open_set, compute_key(node) + (node,))

    def process_state():
        while open_set:
            current_key, current_node = heapq.heappop(open_set)[:2]

            # If the node is in the open set and its key is valid, continue processing
            if current_key < compute_key(current_node):
                heapq.heappush(open_set, (current_key, current_node))
                return

            if g_score[current_node] > rhs[current_node]:
                g_score[current_node] = rhs[current_node]
                for neighbor in G.neighbors(current_node):
                    update_vertex(neighbor)
            else:
                g_score[current_node] = float('inf')
                update_vertex(current_node)

    # Main loop to process the nodes
    while g_score[target] > rhs[target]:
        process_state()

    # Reconstruct the path from source to target
    path = []
    current = source
    while current != target:
        path.append(current)
        next_node = min(G.neighbors(current), key=lambda nbr: g_score[nbr] + G[current][nbr].get(weight, float('inf')))
        current = next_node

    path.append(target)
    return path


# 7
def a_star_search(G, source, target, weight="weight"):
    """
    A* Search Algorithm for finding the shortest path in a weighted graph.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute that holds the weight (default is "weight")

    Returns: the shortest path from source to target and the cost of the path.
    """

    # Heuristic function (in this case, we use a simple zero heuristic, can be customized)
    def heuristic(u, v):
        return 0  # Zero heuristic makes A* equivalent to Dijkstra's algorithm

    # Priority queue for nodes to explore (min-heap), initialized with the source
    priority_queue = [(0, source)]  # (cost, node)

    # Dictionary to store the cost to reach each node
    g_cost = {source: 0}  # Cost from start to node

    # Dictionary to store the parent of each node to reconstruct the path
    came_from = {source: None}

    # Dictionary to store the total estimated cost (g + heuristic) to reach the target
    f_cost = {source: heuristic(source, target)}

    while priority_queue:
        # Get the node with the lowest f_cost (cost + heuristic)
        current_cost, current_node = heapq.heappop(priority_queue)

        # If we reached the target, reconstruct the path and return it
        if current_node == target:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            path.reverse()  # Reverse the path since we built it backwards
            return path, g_cost[target]

        # Explore neighbors of the current node
        for neighbor in G.neighbors(current_node):
            tentative_g_cost = g_cost[current_node] + G[current_node][neighbor].get(weight, 1)

            # Only consider this path if it is better than any previous one
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                f_cost[neighbor] = tentative_g_cost + heuristic(neighbor, target)
                came_from[neighbor] = current_node

                # Add the neighbor to the priority queue with the new estimated cost
                heapq.heappush(priority_queue, (f_cost[neighbor], neighbor))

    return None, float('inf')  # Return None if no path is found


# 8
def monte_carlo_tree_search(G, source, target, weight="weight", iterations=1000, exploration_weight=1.414):
    """
    Monte Carlo Tree Search (MCTS) algorithm to find the shortest path in a weighted graph.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute that holds the weight (default is "weight")
    iterations: number of iterations to run MCTS
    exploration_weight: parameter to balance exploration and exploitation

    Returns: the best path found from source to target and its total cost.
    """

    class MCTSNode:
        def __init__(self, state, parent=None):
            self.state = state
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0

        def expand(self):
            """Expand by adding a child node for each neighboring node."""
            neighbors = list(G.neighbors(self.state))
            for neighbor in neighbors:
                if all(child.state != neighbor for child in self.children):
                    child_node = MCTSNode(neighbor, parent=self)
                    self.children.append(child_node)

        def is_fully_expanded(self):
            """Check if the node has been fully expanded."""
            neighbors = list(G.neighbors(self.state))
            return len(self.children) == len(neighbors)

        def best_child(self, exploration_weight=exploration_weight):
            """Select the child node with the highest Upper Confidence Bound (UCB1)."""
            choices_weights = []
            for child in self.children:
                if child.visits == 0:
                    # Encourage exploration of unvisited nodes
                    choice_weight = float('inf')
                else:
                    choice_weight = (child.value / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                choices_weights.append(choice_weight)

            # Return the child with the highest UCB1 value
            return self.children[choices_weights.index(max(choices_weights))]

        def rollout(self):
            """Simulate a random path to the target (randomly walk to goal) and return its total cost."""
            current_node = self.state
            total_cost = 0
            while current_node != target:
                neighbors = list(G.neighbors(current_node))
                if not neighbors:
                    return float('inf')  # If there are no neighbors, consider it an invalid path
                next_node = random.choice(neighbors)
                total_cost += G[current_node][next_node].get(weight, 1)
                current_node = next_node
            return total_cost

        def backpropagate(self, result):
            """Propagate the result up the tree, updating value and visit count."""
            self.visits += 1
            self.value += result
            if self.parent:
                self.parent.backpropagate(result)

    # Step 1: Initialize root node
    root = MCTSNode(source)

    for _ in range(iterations):
        # Step 2: Select a promising node using UCB1 formula
        node = root
        while not node.is_fully_expanded() and node.children:
            node = node.best_child(exploration_weight)

        # Step 3: Expand the node if it has not been expanded fully
        if not node.is_fully_expanded():
            node.expand()

        # Step 4: Simulate a random path (rollout)
        cost = node.rollout()

        # Step 5: Backpropagate the result up the tree
        node.backpropagate(cost)

    # Step 6: Return the best path found
    best_node = root.best_child(exploration_weight=0)  # Exploitation (no exploration)
    path = []
    current_node = best_node
    while current_node:
        path.append(current_node.state)
        current_node = current_node.parent
    path.reverse()  # Start from source to target
    best_cost = best_node.value / best_node.visits if best_node.visits > 0 else float('inf')

    return path, best_cost


# 9
def yen_k_shortest_paths(G, source, target, K, weight="weight"):
    """
    Yen’s K-Shortest Paths Algorithm to find the K shortest paths in a weighted graph.

    G: networkx graph
    source: start node
    target: goal node
    K: number of shortest paths to find
    weight: edge attribute that holds the weight (default is "weight")

    Returns: A list of the K shortest paths, each with its total cost.
    """

    def dijkstra_shortest_path(G, source, target, weight="weight"):
        """ Helper function to find the shortest path using Dijkstra's algorithm. """
        return nx.shortest_path(G, source=source, target=target, weight=weight), \
            nx.shortest_path_length(G, source=source, target=target, weight=weight)

    # Step 1: Find the shortest path from source to target
    shortest_path, cost = dijkstra_shortest_path(G, source, target, weight)
    if not shortest_path:
        return []  # If no path is found, return an empty list

    # List to store the K shortest paths
    k_shortest_paths = [(shortest_path, cost)]

    # Priority queue to store potential k-th shortest paths
    potential_paths = []

    for k in range(1, K):
        for i in range(len(k_shortest_paths[k - 1][0]) - 1):
            # Step 2: Define the spur node and root path
            spur_node = k_shortest_paths[k - 1][0][i]
            root_path = k_shortest_paths[k - 1][0][:i + 1]

            # Temporarily remove the edges that connect the root path to the rest of the graph
            removed_edges = []
            for path, _ in k_shortest_paths:
                if root_path == path[:i + 1]:
                    u, v = path[i], path[i + 1]
                    if G.has_edge(u, v):
                        removed_edges.append((u, v, G[u][v][weight]))
                        G.remove_edge(u, v)

            # Step 3: Find the spur path from the spur node to the target
            try:
                spur_path, spur_cost = dijkstra_shortest_path(G, spur_node, target, weight)
                total_path = root_path[:-1] + spur_path  # Combine root path and spur path
                total_cost = nx.path_weight(G, total_path, weight)
                heapq.heappush(potential_paths, (total_cost, total_path))
            except nx.NetworkXNoPath:
                # If there's no path from the spur node, skip it
                pass

            # Step 4: Restore the removed edges
            for u, v, w in removed_edges:
                G.add_edge(u, v, **{weight: w})

        # Step 5: Add the lowest cost potential path to k_shortest_paths
        if potential_paths:
            cost, path = heapq.heappop(potential_paths)
            k_shortest_paths.append((path, cost))
        else:
            break  # No more paths available

    return k_shortest_paths


# 10
def ant_colony_optimization(G, source, target, weight="weight", alpha=1.0, beta=2.0, evaporation_rate=0.5, ant_count=10, iterations=100):
    """
    Ant Colony Optimization (ACO) algorithm to find the shortest path in a weighted graph.

    G: networkx graph
    source: start node
    target: goal node
    weight: edge attribute that holds the weight (default is "weight")
    alpha: influence of pheromone trails
    beta: influence of heuristic (edge weight)
    evaporation_rate: rate at which pheromones evaporate
    ant_count: number of ants in each iteration
    iterations: number of iterations to run the algorithm

    Returns: the shortest path from source to target and its total cost.
    """

    # Initialize pheromone levels on each edge (set to a small positive number)
    pheromones = {edge: 1.0 for edge in G.edges()}

    # Helper function to calculate the heuristic, which is 1 / distance (weight) in this case
    def heuristic(u, v):
        return 1.0 / G[u][v].get(weight, 1.0)

    # Function for an ant to build a path from source to target
    def build_path():
        path = [source]
        total_cost = 0

        while path[-1] != target:
            current_node = path[-1]
            neighbors = list(G.neighbors(current_node))

            if not neighbors:
                return None, float('inf')  # If no neighbors, no valid path

            # Calculate probabilities for each neighbor based on pheromone and heuristic
            probabilities = []
            total_prob = 0.0
            for neighbor in neighbors:
                edge = (current_node, neighbor) if (current_node, neighbor) in pheromones else (neighbor, current_node)
                pheromone_level = pheromones[edge]
                heuristic_value = heuristic(current_node, neighbor)
                prob = (pheromone_level ** alpha) * (heuristic_value ** beta)
                probabilities.append((prob, neighbor))
                total_prob += prob

            # Normalize the probabilities
            probabilities = [(prob / total_prob, neighbor) for prob, neighbor in probabilities]

            # Choose the next node based on the calculated probabilities (roulette wheel selection)
            rand = random.random()
            cumulative_prob = 0.0
            next_node = None
            for prob, neighbor in probabilities:
                cumulative_prob += prob
                if rand <= cumulative_prob:
                    next_node = neighbor
                    break

            if next_node is None:
                return None, float('inf')  # If no valid node is chosen

            path.append(next_node)
            total_cost += G[current_node][next_node].get(weight, 1)

        return path, total_cost

    # Function to update pheromones on the graph based on the ants' paths
    def update_pheromones(ant_paths):
        # Evaporate pheromones
        for edge in pheromones:
            pheromones[edge] *= (1.0 - evaporation_rate)

        # Add new pheromones based on the ants' paths
        for path, cost in ant_paths:
            pheromone_deposit = 1.0 / cost  # More pheromone for shorter paths
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1]) if (path[i], path[i + 1]) in pheromones else (path[i + 1], path[i])
                pheromones[edge] += pheromone_deposit

    # Main loop for the ACO algorithm
    best_path = None
    best_cost = float('inf')

    for _ in range(iterations):
        # Step 1: Deploy ants and collect their paths
        ant_paths = []
        for _ in range(ant_count):
            path, cost = build_path()
            if path is not None and cost < best_cost:
                best_path = path
                best_cost = cost
            ant_paths.append((path, cost))

        # Step 2: Update pheromones based on the collected paths
        update_pheromones(ant_paths)

    return best_path, best_cost


# Function to run an algorithm and return its result and execution time
def run_algorithm(algorithm_func, G, source, target, weight='weight', iterations=1000, updated_edges=None):
    # Ensure we copy the graph for each algorithm execution (to avoid shared state issues)
    G_copy = G.copy()

    # Handle algorithms that need extra parameters
    if algorithm_func == dynamic_shortest_path and updated_edges:
        return algorithm_func(G_copy, source, target, weight, updated_edges)

    if algorithm_func == monte_carlo_tree_search:
        return algorithm_func(G_copy, source, target, weight, iterations)

    if algorithm_func == ant_colony_optimization:
        return algorithm_func(G_copy, source, target, weight, iterations=iterations)

    if algorithm_func == yen_k_shortest_paths:
        return algorithm_func(G_copy, source, target, K=1, weight=weight)

    # For others, directly return the result
    return algorithm_func(G_copy, source, target, weight)

# Parallel execution with joblib
def parallel_execution(algorithms, G, source, target, weight="weight", updated_edges=None, iterations=1000):
    results = Parallel(n_jobs=-1)(
        delayed(run_algorithm)(algorithm_func, G, source, target, weight, iterations, updated_edges)
        for algorithm_func in algorithms
    )
    return results

# Example use case
def main():
    # Example graph, you can replace it with your actual graph generation
    G = nx.gnp_random_graph(10, 0.3)  # Smaller graph for demo purposes
    source, target = 0, 9  # Define source and target

    # List of algorithms (include your old ones + new ones like D* Lite, A*, MCTS, etc.)
    algorithms = [
        dijkstra_search,
        floyd_warshall_search,
        bellman_ford_search,
        bidirectional_search,
        dynamic_shortest_path,
        #d_star_lite,   # takes long to process (haven't reached output yet)
        a_star_search,
        monte_carlo_tree_search,
        yen_k_shortest_paths,
        ant_colony_optimization
    ]

    # Parallel execution
    results = parallel_execution(algorithms, G, source, target)

    # Print results
    for i, result in enumerate(results):
        print(f"Algorithm {i+1} result: {result}")

if __name__ == "__main__":
    main()
