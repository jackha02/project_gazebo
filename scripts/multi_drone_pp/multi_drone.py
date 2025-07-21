import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
import networkx as nx
import xml.etree.ElementTree as ET

from path_planning import (
    load_path_points_from_csv,
    dijkstra_path,
    insert_detour_points_collision_aware,
    plot_cylinder
)

from inspection_points import (
    parse_sdf_with_visual_pose,
    sample_points_on_cylinders,
    create_buffered_cylinder,
    project_to_buffered_cylinder,
    find_path_points,
    visualize_all,
    create_buffered_cylinder,
    BUFFER
)

# === Multi-Drone Setup ===
NUM_DRONES = 3  # Change to 1, 2, 3, or 4
PENALTY_LAMBDA = 50.0  # Collision penalty for Dijkstra

# Define individual starting points
DRONE_1_START = [0.0, 0.0, 1.5]
DRONE_2_START = [10.0, 0.0, 1.5]
DRONE_3_START = [0.0, 10.0, 1.5]
DRONE_4_START = [10.0, 10.0, 1.5]

# Collect only the starting points needed
STARTING_POINTS = []
if NUM_DRONES >= 1: STARTING_POINTS.append([DRONE_1_START])
if NUM_DRONES >= 2: STARTING_POINTS.append([DRONE_2_START])
if NUM_DRONES >= 3: STARTING_POINTS.append([DRONE_3_START])
if NUM_DRONES >= 4: STARTING_POINTS.append([DRONE_4_START])

assert len(STARTING_POINTS) == NUM_DRONES, "Mismatch in number of starting points and drones."

# === Input Files ===
csv_path = os.path.join("..", "output", "inspection_points.csv")
sdf_file = os.path.join("..", "..", "models", "full_jacket", "model.sdf")

# === Step 1: Load inspection points ===
inspection_points = load_path_points_from_csv(csv_path)

# === Step 2: Parse cylinder model + buffer ===
cylinders = parse_sdf_with_visual_pose(sdf_file)
buffered_cylinders = [create_buffered_cylinder(cyl, BUFFER) for cyl in cylinders]

# === Step 3: Assign clusters to drones ===

# 1. KMeans clustering with start-position-based centroids
init_centroids = np.array([start[0] for start in STARTING_POINTS])
kmeans = KMeans(n_clusters=NUM_DRONES, init=init_centroids, n_init=1, random_state=0)
assignments = kmeans.fit_predict(np.array(inspection_points))

# 2. Group points into clusters
clusters = [[] for _ in range(NUM_DRONES)]
for point, label in zip(inspection_points, assignments):
    clusters[label].append(point)

# 3. Estimate total Dijkstra path length for each cluster
cluster_path_costs = []
for cluster in clusters:
    cluster_path = dijkstra_path(cluster, cylinders, penalty_lambda=PENALTY_LAMBDA)
    total_length = sum(np.linalg.norm(np.array(cluster_path[i]) - np.array(cluster_path[i+1])) for i in range(len(cluster_path) - 1))
    cluster_path_costs.append(total_length)

# 4. Build cost matrix [drone i][cluster j] = start_dist + estimated path cost
assignment_matrix = np.zeros((NUM_DRONES, NUM_DRONES))
for i, drone_start in enumerate([p[0] for p in STARTING_POINTS]):
    for j in range(NUM_DRONES):
        start_to_cluster = np.linalg.norm(np.mean(clusters[j], axis=0) - np.array(drone_start))
        assignment_matrix[i][j] = 0.5 * start_to_cluster + cluster_path_costs[j]

# 5. Optimal assignment using Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(assignment_matrix)

# 6. Assign clusters to drones
assigned_clusters = [None] * NUM_DRONES
drone_to_cluster = {}

def compute_cluster_cost(cluster, start, cylinders, buffered_cylinders, return_weight=2.0, detour_weight=5.0):
    path = dijkstra_path([start] + cluster, cylinders, penalty_lambda=PENALTY_LAMBDA)
    path = insert_detour_points_collision_aware(path, buffered_cylinders)

    if not path or len(path) < 2:
        return float("inf"), []

    path_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i+1])) for i in range(len(path)-1))
    return_offset = np.linalg.norm(np.array(path[-1]) - np.array(start))

    # Count detour points = points not in [start] + cluster
    base_set = set(tuple(np.round(p, 3)) for p in [start] + cluster)
    detour_points = [p for p in path if tuple(np.round(p, 3)) not in base_set]
    num_detours = len(detour_points)

    total_cost = path_length + return_weight * return_offset + detour_weight * num_detours

    return total_cost, path

def check_path_overlap(path1, path2, buffer=1.0):
    for p1 in path1:
        for p2 in path2:
            if np.linalg.norm(np.array(p1) - np.array(p2)) < buffer:
                return True
    return False

def rebalance_clusters(clusters, starts, cylinders, buffered_cylinders, max_iter=20):
    for _ in range(max_iter):
        costs = []
        paths = []
        for i in range(NUM_DRONES):
            cost, path = compute_cluster_cost(clusters[i], starts[i][0], cylinders, buffered_cylinders)
            costs.append(cost)
            paths.append(path)

        max_id = np.argmax(costs)
        min_id = np.argmin(costs)
        if costs[max_id] - costs[min_id] < 1.0:
            break  # Balanced enough

        moved = False
        for pt in clusters[max_id][:]:
            trial_max = clusters[max_id][:]
            trial_min = clusters[min_id][:]
            trial_max.remove(pt)
            trial_min.append(pt)

            c1, p1 = compute_cluster_cost(trial_max, starts[max_id][0], cylinders, buffered_cylinders)
            c2, p2 = compute_cluster_cost(trial_min, starts[min_id][0], cylinders, buffered_cylinders)

            # Ensure no overlap
            overlap_free = True
            for i, path_i in enumerate(paths):
                if i not in [max_id, min_id]:
                    if check_path_overlap(p1, path_i, buffer=1.0) or check_path_overlap(p2, path_i, buffer=1.0):
                        overlap_free = False
                        break

            if overlap_free and abs(c1 - c2) < abs(costs[max_id] - costs[min_id]):
                clusters[max_id] = trial_max
                clusters[min_id] = trial_min
                moved = True
                break  # Accept first improving move

        if not moved:
            break  # No better move found

    return clusters

for drone_id, cluster_idx in zip(row_ind, col_ind):
    assigned_clusters[drone_id] = clusters[cluster_idx]
    drone_to_cluster[drone_id] = cluster_idx

# === Step 4: Path planning + export ===

# Rebalance cluster assignments for balanced cost & disjoint paths
assigned_clusters = rebalance_clusters(assigned_clusters, STARTING_POINTS, cylinders, buffered_cylinders)

final_paths = []
for drone_id in range(NUM_DRONES):
    drone_path = STARTING_POINTS[drone_id] + assigned_clusters[drone_id]
    sorted_path = dijkstra_path(drone_path, cylinders, penalty_lambda=PENALTY_LAMBDA)
    full_path = insert_detour_points_collision_aware(sorted_path, buffered_cylinders)
    final_paths.append((full_path, drone_path))

    # Save with correct drone ID
    out_csv = os.path.join("..", "output", f"ipp_drone_{drone_id+1}.csv")
    pd.DataFrame(full_path, columns=["x", "y", "z"]).to_csv(out_csv, index=False)
    
print("\n[Checking inter-drone path overlaps]")
for i in range(NUM_DRONES):
    for j in range(i + 1, NUM_DRONES):
        path_i = final_paths[i][0]
        path_j = final_paths[j][0]
        if check_path_overlap(path_i, path_j, buffer=1.0):
            print(f"⚠️  Path overlap detected between Drone {i+1} and Drone {j+1}")
        else:
            print(f" Success!")

# === Step 5: Visualization ===
colors = ['blue', 'red', 'green', 'purple']  # Add more if needed

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw cylinders and buffered zones
for cyl in cylinders:
    plot_cylinder(ax, cyl, color='gray', alpha=0.5)
    plot_cylinder(ax, create_buffered_cylinder(cyl, BUFFER), color='cyan', alpha=0.3)

# Draw drone paths
for drone_id, (full_path, orig_path) in enumerate(final_paths):
    color = colors[drone_id % len(colors)]
    arr = np.array(full_path)
    ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, label=f'Drone {drone_id+1} Path')

    orig_set = set(tuple(p) for p in orig_path)
    detour_pts = [p for p in full_path if tuple(p) not in orig_set]
    normal_pts = [p for p in full_path if tuple(p) in orig_set]

    if normal_pts:
        normal_arr = np.array(normal_pts)
        ax.scatter(normal_arr[:, 0], normal_arr[:, 1], normal_arr[:, 2],
                   c=color, s=40, marker='o', label=f'Drone {drone_id+1} Points')

    if detour_pts:
        detour_arr = np.array(detour_pts)
        ax.scatter(detour_arr[:, 0], detour_arr[:, 1], detour_arr[:, 2],
                   c='lime', s=60, marker='x', label=f'Drone {drone_id+1} Detours')

    # Mark starting point
    start = STARTING_POINTS[drone_id][0]
    ax.scatter(start[0], start[1], start[2],
               c=color, s=80, marker='^', label=f'Drone {drone_id+1} Start')

# Axes and title
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Multi-Drone Inspection Paths')
ax.legend()
ax.view_init(elev=25, azim=35)
plt.tight_layout()
plt.show()

