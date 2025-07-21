import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation as R
import networkx as nx
import heapq
import xml.etree.ElementTree as ET

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

# Returns the order of inspection points using Dijkstra's path optimization
def dijkstra_path(points, cylinders, penalty_lambda=15.0):
    G = nx.Graph()
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                distance = euclidean(p1, p2)
                num_collisions = sum(len(ray_cylinder_intersection(p1, p2, cyl)) for cyl in cylinders)
                cost = distance + penalty_lambda * num_collisions
                G.add_edge(i, j, weight=cost)
    
    path_indices = [0]
    visited = set(path_indices)
    while len(visited) < len(points):
        last = path_indices[-1]
        neighbors = [(j, G[last][j]['weight']) for j in G.neighbors(last) if j not in visited]
        if not neighbors:
            break
        next_node = min(neighbors, key=lambda x: x[1])[0]
        path_indices.append(next_node)
        visited.add(next_node)

    return [points[i] for i in path_indices]

# Collision avoidance
def ray_cylinder_intersection(A, C, cyl):
    A = np.array(A)
    C = np.array(C)

    P = cyl["position"]
    R = cyl["rotation"]
    r = cyl["radius"]
    h = cyl["length"]

    R_inv = R.T
    A_local = R_inv @ (A - P)
    C_local = R_inv @ (C - P)
    d = C_local - A_local

    hits = []

    # Side surface intersection
    a = d[0]**2 + d[1]**2
    b = 2 * (A_local[0]*d[0] + A_local[1]*d[1])
    c = A_local[0]**2 + A_local[1]**2 - r**2

    if a != 0:
        disc = b**2 - 4*a*c
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            for sign in [-1, 1]:
                t = (-b + sign * sqrt_disc) / (2 * a)
                if 0 < t < 1:
                    z = A_local[2] + t * d[2]
                    if -h/2 <= z <= h/2:
                        pt_local = A_local + t * d
                        pt_world = R @ pt_local + P
                        hits.append((t, pt_world))

    # Cap intersection
    for z_cap in [-h/2, h/2]:
        if d[2] != 0:
            t = (z_cap - A_local[2]) / d[2]
            if 0 < t < 1:
                x = A_local[0] + t * d[0]
                y = A_local[1] + t * d[1]
                if x**2 + y**2 <= r**2:
                    pt_local = A_local + t * d
                    pt_world = R @ pt_local + P
                    hits.append((t, pt_world))

    if not hits:
        return []

    # --- Cluster based on 0.5 diameter sphere = radius ---
    hits.sort(key=lambda x: x[0])  # sort by t
    clustered = []
    used = [False] * len(hits)

    for i in range(len(hits)):
        if used[i]:
            continue
        t_i, pt_i = hits[i]
        group = [(t_i, pt_i)]
        used[i] = True
        for j in range(i + 1, len(hits)):
            if used[j]:
                continue
            t_j, pt_j = hits[j]
            if np.linalg.norm(pt_i - pt_j) < 2 * 0.5:
                group.append((t_j, pt_j))
                used[j] = True
        group.sort(key=lambda x: x[0])  # sort group by t
        clustered.append(group[0])  # keep only earliest

    # Return only the earliest from all clustered groups
    return [min(clustered, key=lambda x: x[0])]

def is_segment_safe(A, B, cylinders):
    for cyl in cylinders:
        if ray_cylinder_intersection(A, B, cyl):
            return False
    return True

def generate_safe_detour_polyline(A, C, hits, cylinders, max_attempts=500):
    A, C = np.array(A), np.array(C)
    directions = []
    for _, pt, cyl in hits:
        v_path = C - A
        v_path /= np.linalg.norm(v_path)
        cyl_axis = cyl["rotation"] @ np.array([0, 0, 1])
        local_pt = cyl["rotation"].T @ (pt - cyl["position"])
        radial_dir_local = np.array([local_pt[0], local_pt[1], 0.0])
        if np.linalg.norm(radial_dir_local) < 1e-6:
            return None
        radial_dir_local /= np.linalg.norm(radial_dir_local)
        outward_dir_world = cyl["rotation"] @ radial_dir_local
        detour_dir = np.cross(np.cross(outward_dir_world, v_path), v_path)
        detour_dir /= np.linalg.norm(detour_dir)
        directions.append((pt, detour_dir))
    for _ in range(max_attempts):
        detours = []
        valid = True
        for pt, d in directions:
            offset = np.random.uniform(0.5, 3.0)
            B = np.array(pt) + offset * d
            if not is_segment_safe(A, B, cylinders):
                valid = False
                break
            detours.append(B.tolist())
            A = B
        if valid and is_segment_safe(A, C, cylinders):
            return detours
    return None

def insert_detour_points_collision_aware(path, cylinders):
    updated_path = []
    i = 0

    while i < len(path) - 1:
        A = path[i]
        C = path[i + 1]

        if not updated_path or not np.allclose(updated_path[-1], A, atol=1e-6):
            updated_path.append(A)

        current_start = A
        while True:
            hits = []
            for cyl in cylinders:
                intersections = ray_cylinder_intersection(current_start, C, cyl)
                if intersections:
                    t_hit, pt_hit = intersections[0]
                    dist = np.linalg.norm(np.array(pt_hit) - np.array(current_start))
                    hits.append((dist, pt_hit, cyl))

            if not hits:
                break

            # Sort by 3D distance from current_start (not t)
            hits.sort(key=lambda x: x[0])

            # Debug print: show hits in ascending order of distance
            for dist, pt, _ in hits:
                print(f"    at distance={dist:.4f}, point={pt}")

            _, first_hit_point, hit_cylinder = hits[0]

            detour_polyline = generate_safe_detour_polyline(current_start, C, hits, cylinders)
            if not detour_polyline:
                print(f"[Warning] Could not offset detour from {current_start} to {C}")
                break

            for point in detour_polyline:
                if not np.allclose(updated_path[-1], point, atol=1e-6):
                    updated_path.append(point)

            current_start = detour_polyline[-1]

        if not np.allclose(updated_path[-1], C, atol=1e-6):
            updated_path.append(C)

        i += 1

    return updated_path

# Parse sdf
def parse_sdf_for_full_generation(sdf_path):
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    cylinders = []

    for link in root.iter("link"):
        pose_elem = link.find("pose")
        visual_elem = link.find("visual")

        if pose_elem is None or visual_elem is None:
            continue

        pose_vals = list(map(float, pose_elem.text.strip().split()))
        link_position = np.array(pose_vals[:3])
        link_rotation = R.from_euler('xyz', pose_vals[3:])

        geometry_elem = visual_elem.find("geometry")
        if geometry_elem is None:
            continue

        cylinder_elem = geometry_elem.find("cylinder")
        if cylinder_elem is None:
            continue

        radius = float(cylinder_elem.find("radius").text.strip())
        length = float(cylinder_elem.find("length").text.strip())

        visual_pose = np.zeros(6)
        visual_pose_elem = visual_elem.find("pose")
        if visual_pose_elem is not None:
            visual_pose = list(map(float, visual_pose_elem.text.strip().split()))

        visual_position = np.array(visual_pose[:3]) 
        visual_rotation = R.from_euler('xyz', visual_pose[3:])

        world_rot = (link_rotation * visual_rotation).as_matrix()
        position = link_position + link_rotation.apply(visual_position)

        if radius > 0 and length > 0:
            cylinders.append({
                "position": position,
                "rotation": world_rot,
                "radius": radius,
                "length": length
            })

    return cylinders
    
# CSV Export
def load_path_points_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df[['x', 'y', 'z']].values.tolist()

def plot_cylinder(ax, cyl, color='gray', alpha=1.0, resolution=15):
    z = np.linspace(-cyl['length']/2, cyl['length']/2, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = cyl['radius'] * np.cos(theta_grid)
    y_grid = cyl['radius'] * np.sin(theta_grid)

    points = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
    rotated = (cyl['rotation'] @ points.T).T + cyl['position']
    X = rotated[:, 0].reshape(z.shape[0], -1)
    Y = rotated[:, 1].reshape(z.shape[0], -1)
    Z = rotated[:, 2].reshape(z.shape[0], -1)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

def visualize_path_and_model(sdf_file, full_path, inspection_path_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cylinders = parse_sdf_with_visual_pose(sdf_file)
    for cyl in cylinders:
        plot_cylinder(ax, cyl, color='gray', alpha=0.5)
        plot_cylinder(ax, create_buffered_cylinder(cyl, BUFFER), color='cyan', alpha=0.3) 

    path_arr = np.array(full_path)
    ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], color='blue', label='Path')

    # Separate detour points and inspection points
    orig_set = set(tuple(p) for p in inspection_path_points)
    detour_points = [p for p in full_path if tuple(p) not in orig_set]
    normal_points = [p for p in full_path if tuple(p) in orig_set]

    if normal_points:
        normal_arr = np.array(normal_points)
        ax.scatter(normal_arr[:, 0], normal_arr[:, 1], normal_arr[:, 2], c='red', label='Inspection Points', s=40)

    if detour_points:
        detour_arr = np.array(detour_points)
        ax.scatter(detour_arr[:, 0], detour_arr[:, 1], detour_arr[:, 2],
                   c='lime', label='Detour Points', s=60, marker='o')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Model + Path Planning Points')
    ax.legend()
    ax.view_init(elev=25, azim=35)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_path = os.path.join("..", "output", "inspection_points.csv")
    sdf_file = os.path.join("..", "..", "models", "full_jacket", "model.sdf")
    
    # Step 1: Load the cylinder model and generate inspection points
    inspection_path_points = load_path_points_from_csv(csv_path)
    
    # Step 2: Create buffered boundaries for each cylinder
    cylinders = parse_sdf_for_full_generation(sdf_file)
    buffered_cylinders = [create_buffered_cylinder(cyl, BUFFER) for cyl in cylinders]
    
    # Step 3: Compute Dijkstra path from inspection points
    sorted_path = dijkstra_path(inspection_path_points, cylinders)

    capsules = []
    for cyl in buffered_cylinders:
    	axis = cyl["rotation"] @ np.array([0, 0, 1])
    	start = cyl["position"] - axis * cyl["length"] / 2
    	end = cyl["position"] + axis * cyl["length"] / 2
    	cap_min = np.minimum(start, end) - cyl["radius"]
    	cap_max = np.maximum(start, end) + cyl["radius"]
    	capsules.append({"aabb": (cap_min, cap_max)})

    # Step 4: Insert detour points if collision is detected
    full_path = insert_detour_points_collision_aware(sorted_path, buffered_cylinders)

    # Step 5: Export final collision-aware path
    output_csv = os.path.join("..", "output", "path_planning.csv")
    pd.DataFrame(full_path, columns=["x", "y", "z"]).to_csv(output_csv, index=False)

    # Step 6: Visualize the result
    visualize_path_and_model(sdf_file, full_path, inspection_path_points)

# Check for intersection points along each segment
if __name__ == "__main__":
    ...
    # Check for intersection points along each segment
    total_hits = 0
    for i in range(len(sorted_path) - 1):
        A = sorted_path[i]
        C = sorted_path[i + 1]

        segment_hits = []
        for cyl in buffered_cylinders:
            hits = ray_cylinder_intersection(A, C, cyl)
            if hits:
                t, pt = hits[0]
                segment_hits.append((t, pt, cyl))

        # Sort hits by t
        segment_hits.sort(key=lambda x: x[0])

        if segment_hits:
            print(f"[Hit] {len(segment_hits)} intersection(s) for segment {A} → {C}")
            for t, pt, _ in segment_hits:
                print(f"    at t={t:.4f}, point={pt}")
                total_hits += 1

    print(f"Total intersection points found: {total_hits}")


