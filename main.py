import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import time
import random

def create_graph(points, metric='euclidean'):
    dist_matrix = squareform(pdist(points, metric=metric))
    return dist_matrix

def minimum_spanning_tree(graph):
    num_vertices = graph.shape[0]
    visited = np.zeros(num_vertices, dtype=bool)
    mst_edges = []
    visited[0] = True
    for _ in range(num_vertices - 1):
        min_edge = float('inf')
        a, b = -1, -1
        for i in range(num_vertices):
            if visited[i]:
                for j in range(num_vertices):
                    if not visited[j] and graph[i, j] > 0 and graph[i, j] < min_edge:
                        min_edge = graph[i, j]
                        a, b = i, j
        mst_edges.append((a, b))
        visited[b] = True
    mst = np.zeros_like(graph)
    for a, b in mst_edges:
        mst[a, b] = mst[b, a] = graph[a, b]
    return mst

def find_odd_degree_vertices(mst):
    degree = np.sum(mst > 0, axis=0)
    return np.where(degree % 2 == 1)[0]

def eulerian_circuit(mst, start):
    edges = []
    for i in range(len(mst)):
        for j in range(len(mst)):
            if mst[i, j] > 0:
                edges.append((i, j))
    circuit = []
    stack = [start]
    while stack:
        u = stack[-1]
        found = False
        for v in range(len(mst)):
            if mst[u, v] > 0:
                stack.append(v)
                mst[u, v] -= 1
                mst[v, u] -= 1
                found = True
                break
        if not found:
            circuit.append(stack.pop())
    return circuit

def shortcut_tour(tour):
    visited = set()
    shortcut = []
    for vertex in tour:
        if vertex not in visited:
            shortcut.append(vertex)
            visited.add(vertex)
    return shortcut

def christofides(points, metric):
    graph = create_graph(points, metric)
    mst = minimum_spanning_tree(graph)
    odd_vertices = find_odd_degree_vertices(mst)
    odd_graph = graph[np.ix_(odd_vertices, odd_vertices)]
    matching = min_weight_matching(odd_graph, odd_vertices)
    multigraph = np.zeros_like(graph)
    for a, b in matching:
        multigraph[a, b] = multigraph[b, a] = graph[a, b]
    for i, j in zip(*np.nonzero(mst)):
        multigraph[i, j] = multigraph[j, i] = mst[i, j]
    start_vertex = odd_vertices[0]
    euler_circuit = eulerian_circuit(multigraph, start_vertex)
    tour = shortcut_tour(euler_circuit)
    return tour

def min_weight_matching(odd_graph, odd_vertices):
    num_odd = odd_graph.shape[0]
    matched = np.zeros(num_odd, dtype=bool)
    matching = []
    for i in range(num_odd):
        if matched[i]:
            continue
        min_weight = float('inf')
        min_j = -1
        for j in range(num_odd):
            if not matched[j] and i != j and odd_graph[i, j] < min_weight:
                min_weight = odd_graph[i, j]
                min_j = j
        if min_j != -1:
            matching.append((odd_vertices[i], odd_vertices[min_j]))
            matched[i] = matched[min_j] = True
    return matching

def plot_tour(points, tour, show_mst=False, mst=None):
    plt.figure(figsize=(10, 8))
    plt.plot(points[:, 0], points[:, 1], 'bo', label='Points')
    if show_mst and mst is not None:
      mst_plotted = False
      for i, j in zip(*np.nonzero(mst)):
          if not mst_plotted:
              plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]],
                      'g-', linewidth=2, label='MST')
              mst_plotted = True
          else:
              plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]],
                      'g-', linewidth=2)

    for i in range(len(tour) - 1):
        plt.plot([points[tour[i], 0], points[tour[i + 1], 0]], [points[tour[i], 1], points[tour[i + 1], 1]], 'r-', linewidth=2)
    plt.plot([points[tour[-1], 0], points[tour[0], 0]], [points[tour[-1], 1], points[tour[0], 1]], 'r--')
    plt.title("Christofides Algorithm Tour")
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid()
    plt.legend()
    plt.show()

def save_tour(tour, points, file_name='tour_data.csv'):
    tour_data = pd.DataFrame({
        'Point': [f'P{i}' for i in tour],
        'X': points[tour, 0],
        'Y': points[tour, 1]
    })
    tour_data.to_csv(file_name, index=False)
    print(f"Tour data saved to '{file_name}'.")

def validate_input(x_points, y_points):
    try:
        x_vals = list(map(float, x_points.split(',')))
        y_vals = list(map(float, y_points.split(',')))
        if len(x_vals) != len(y_vals):
            raise ValueError("X and Y coordinates must have the same length.")
        return np.array(x_vals), np.array(y_vals)
    except ValueError as e:
        print(f"Input error: {e}")
        return None, None

def generate_random_points(num_points):
    return np.random.rand(num_points, 2) * 100

def compute_total_distance(tour, points):
    total_distance = 0
    for i in range(len(tour)):
        start = points[tour[i]]
        end = points[tour[(i + 1) % len(tour)]]
        total_distance += np.linalg.norm(start - end)
    return total_distance

def update_plot(x_points, y_points, metric, show_mst):
    points = np.array(list(zip(x_points, y_points)))
    tour = christofides(points, metric)
    mst = minimum_spanning_tree(create_graph(points, metric))
    plot_tour(points, tour, show_mst, mst)
    save_tour(tour, points)
    total_distance = compute_total_distance(tour, points)
    print(f"Total distance of the tour: {total_distance:.2f}")

def on_button_click(b):
    x_vals, y_vals = validate_input(x_points.value, y_points.value)
    if x_vals is not None and y_vals is not None:
        update_plot(x_vals, y_vals, distance_metric.value, show_mst.value)

def on_random_points_button_click(b):
    num_points = int(num_random_points.value)
    random_points = generate_random_points(num_points)
    x_points.value = ','.join(map(str, random_points[:, 0]))
    y_points.value = ','.join(map(str, random_points[:, 1]))
    update_plot(random_points[:, 0], random_points[:, 1], distance_metric.value, show_mst.value)
x_points = widgets.Text(
    description='X Points',
    placeholder='Comma-separated X coordinates'
)

y_points = widgets.Text(
    description='Y Points',
    placeholder='Comma-separated Y coordinates'
)

distance_metric = widgets.Dropdown(
    options=['Euclidean', 'Cityblock'],
    value='Cityblock',
    description='Distance Metric:'
)

show_mst = widgets.Checkbox(
    value=False,
    description='Show MST'
)

button = widgets.Button(description="Calculate Tour")
button.on_click(on_button_click)
num_random_points = widgets.IntSlider(
    value=5,
    min=1,
    max=20,
    step=1,
    description='Number Of Random Points:',
    continuous_update=False
)

random_points_button = widgets.Button(description="Generate Random Points")
random_points_button.on_click(on_random_points_button_click)
help_text = widgets.Label(
    value="Enter the coordinates of points in X and Y as comma-separated values. "
          "Select a distance metric and check options to visualize the MST."
)
display(help_text, x_points, y_points, distance_metric, show_mst, button, num_random_points, random_points_button)
