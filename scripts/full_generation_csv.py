import xml.etree.ElementTree as ET
import pandas as pd
import math
import os
from itertools import permutations
from collections import defaultdict
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.spatial.transform import Rotation as R

def tsp_heuristic(points, start=(0.0, 0.0, 0.0)):
    if not points:
        return []
    current = start
    path = []
    unvisited = points.copy()
    while unvisited:
        next_point = min(unvisited, key=lambda p: euclidean(current, p))
        path.append(next_point)
        unvisited.remove(next_point)
        current = next_point
    return path

def compute_brace_axis(pose, length):
    x, y, z, roll, pitch, yaw = pose
    direction = R.from_euler('xyz', [roll, pitch, yaw]).apply([0, 0, 1])
    start = np.array([x, y, z])
    end = start + direction * length
    return start, end

def closest_point_between_lines(p1, d1, p2, d2):
    w0 = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)
    denominator = a * c - b * b
    if abs(denominator) < 1e-6:
        return (p1 + p2) / 2
    sc = (b * e - c * d) / denominator
    tc = (a * e - b * d) / denominator
    point_on_line1 = p1 + sc * d1
    point_on_line2 = p2 + tc * d2
    midpoint = (point_on_line1 + point_on_line2) / 2
    return midpoint

def brace_intersection_from_rotation(pose1, pose2, length):
    start1, end1 = compute_brace_axis(pose1, length)
    start2, end2 = compute_brace_axis(pose2, length)
    direction1 = end1 - start1
    direction2 = end2 - start2
    return closest_point_between_lines(start1, direction1, start2, direction2)

def connector_joints(link_data):
    return [pose for name, pose in link_data.items() if name.startswith("conn")]

def generate_inspection_points(joints, cx, cy, offset_distance):
    points = []
    for x, y, z in joints:
        dx, dy = x - cx, y - cy
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance == 0:
            dx, dy = 1, 0
        else:
            dx /= distance
            dy /= distance
        insp_x = x + dx * offset_distance
        insp_y = y + dy * offset_distance
        insp_z = z
        points.append((insp_x, insp_y, insp_z))
    return points

def poi_generation(sdf_path: str, output_csv_path: str, offset_distance: float = 3.0):
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    link_data = {}
    brace_groups = defaultdict(list)
    brace_lengths = {}

    for link in root.iter("link"):
        name = link.attrib.get("name", "")
        pose_elem = link.find("pose")
        visual_elem = link.find("visual")

        if pose_elem is not None and visual_elem is not None:
            pose_vals = list(map(float, pose_elem.text.strip().split()))
            x, y, z, roll, pitch, yaw = pose_vals

            geometry_elem = visual_elem.find("geometry")
            if geometry_elem is not None:
                cylinder_elem = geometry_elem.find("cylinder")
                if cylinder_elem is not None:
                    length_elem = cylinder_elem.find("length")
                    if length_elem is not None:
                        length = float(length_elem.text.strip())

                        if name.startswith("conn"):
                            direction = R.from_euler('xyz', [roll, pitch, yaw]).apply([0, 0, 1])
                            offset = direction * (length / 2.0)
                            start_point = np.array([x, y, z]) - offset
                            link_data[name] = tuple(start_point)

                        if name.startswith("xbrace") and any(char.isdigit() for char in name):
                            parts = name.split("_")
                            if len(parts) >= 4:
                                group_key = "_".join(parts[:3])
                                brace_groups[group_key].append((x, y, z, roll, pitch, yaw))
                                brace_lengths[group_key] = length

    brace_inspection_joints = []
    for group_key, poses in brace_groups.items():
        if len(poses) == 2:
            intersection = brace_intersection_from_rotation(poses[0], poses[1], brace_lengths[group_key])
            if intersection is not None:
                brace_inspection_joints.append(tuple(intersection))

    connector_inspection_joints = connector_joints(link_data)
    all_joints = connector_inspection_joints + brace_inspection_joints

    cx = sum(pt[0] for pt in all_joints) / len(all_joints)
    cy = sum(pt[1] for pt in all_joints) / len(all_joints)

    connector_inspection_points = generate_inspection_points(connector_inspection_joints, cx, cy, offset_distance)
    brace_inspection_points = generate_inspection_points(brace_inspection_joints, cx, cy, offset_distance)

    all_points = connector_inspection_points + brace_inspection_points
    sorted_points = tsp_heuristic(all_points)

    inspection_points = [
        {"x": x, "y": y, "z": z} for x, y, z in sorted_points
    ]

    df = pd.DataFrame(inspection_points)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    
if __name__ == "__main__":
    sdf_input = os.path.join("..", "models", "full_jacket", "model.sdf")
    output_csv = os.path.join("..", "config", "joints.csv")
    poi_generation(sdf_input, output_csv)
    print(f"Generated {output_csv} from {sdf_input}")
