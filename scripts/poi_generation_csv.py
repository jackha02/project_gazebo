import xml.etree.ElementTree as ET
import pandas as pd
import math
import os
from itertools import permutations
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import numpy as np


def tsp_heuristic(points, start=None):
    """
    Solve TSP using nearest neighbor heuristic.
    If start is None, pick the point with lowest z value.
    """
    if not points:
        return []

    if start is None:
        current = min(points, key=lambda p: p[2])
    else:
        current = start

    path = [current]
    unvisited = points.copy()
    unvisited.remove(current)

    while unvisited:
        next_point = min(unvisited, key=lambda p: euclidean(current, p))
        path.append(next_point)
        unvisited.remove(next_point)
        current = next_point

    return path


def poi_generation(sdf_path: str, output_csv_path: str, offset_distance: float = 3.0):
    """
    Generate a joints.csv file with inspection points offset from joint positions in a model.sdf,
    based on 45-degree diagonal direction from the center of the jacket frame.

    :param sdf_path: Path to the model.sdf file
    :param output_csv_path: Path to save the generated CSV file
    :param offset_distance: Distance to place the inspection point from each joint (in meters)
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    
    # Find pose and cylinder length of jacket
    joints = []
    for link in root.iter("link"):
        name = link.attrib.get("name", "")
        if not name.startswith("leg") or not any(char.isdigit() for char in name):
            continue

        pose_elem = link.find("pose")
        visual_elem = link.find("visual")
        if pose_elem is not None and visual_elem is not None:
            pose_vals = list(map(float, pose_elem.text.strip().split()))
            x, y, z = pose_vals[0], pose_vals[1], pose_vals[2]

            geometry_elem = visual_elem.find("geometry")
            if geometry_elem is not None:
                cylinder_elem = geometry_elem.find("cylinder")
                if cylinder_elem is not None:
                    length_elem = cylinder_elem.find("length")
                    if length_elem is not None:
                        length = float(length_elem.text.strip())
                        z += 0.5 * length

            joints.append((x, y, z))

    if len(joints) >= 4:
        lowest_joints = sorted(joints, key=lambda pt: pt[2])[:4]
        cx = sum([pt[0] for pt in lowest_joints]) / 4
        cy = sum([pt[1] for pt in lowest_joints]) / 4
        sorted_joints = tsp_heuristic(lowest_joints)
    elif joints:
        cx = sum([pt[0] for pt in joints]) / len(joints)
        cy = sum([pt[1] for pt in joints]) / len(joints)
        sorted_joints = tsp_heuristic(joints)
    else:
        cx, cy = 0, 0
        sorted_joints = []

    inspection_points = []
    for x, y, z in sorted_joints:
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
        inspection_points.append({"x": insp_x, "y": insp_y, "z": insp_z})

    df = pd.DataFrame(inspection_points)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    sdf_input = os.path.join("..", "models", "jacket_frame", "model.sdf")
    output_csv = os.path.join("..", "config", "joints.csv")
    poi_generation(sdf_input, output_csv)
    print(f"Generated {output_csv} from {sdf_input}")

