import xml.etree.ElementTree as ET
import numpy as np
import os
import random
import itertools
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd

BUFFER = 0.1
NUM_INSPECTION_POINTS = 12

def parse_sdf_with_visual_pose(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    links = root.findall('.//link')
    cylinders = []

    for link in links:
        # Get link pose
        pose_tag = link.find('pose')
        link_pose_vals = list(map(float, pose_tag.text.strip().split())) if pose_tag is not None else [0]*6
        link_position = np.array(link_pose_vals[:3])
        link_rotation_rpy = link_pose_vals[3:]
        link_rot = R.from_euler('xyz', link_rotation_rpy)

        geometry = None
        visual_pose_vals = [0]*6  # default if visual pose is not defined

        for tag in ['visual', 'collision']:
            sub = link.find(tag)
            if sub is not None:
                geometry = sub.find('geometry')
                if geometry is not None and geometry.find('cylinder') is not None:
                    # If <pose> under <visual> exists, parse it
                    pose_tag_visual = sub.find('pose')
                    if pose_tag_visual is not None:
                        visual_pose_vals = list(map(float, pose_tag_visual.text.strip().split()))
                    break

        if geometry is None or geometry.find('cylinder') is None:
            continue

        cyl = geometry.find('cylinder')
        radius = float(cyl.find('radius').text)
        length = float(cyl.find('length').text)

        # Compose final pose: link and visual
        visual_position = np.array(visual_pose_vals[:3])
        visual_rotation_rpy = visual_pose_vals[3:]
        visual_rot = R.from_euler('xyz', visual_rotation_rpy)

        # Compose rotation: R_link * R_visual
        world_rot = link_rot * visual_rot
        # Compose position: p_link + R_link * p_visual
        world_position = link_position + link_rot.apply(visual_position)

        cylinders.append({
            'position': world_position,
            'rotation': world_rot.as_matrix(),
            'radius': radius,
            'length': length
        })

    return cylinders

def sample_points_on_cylinders(cylinders, num_points):
    points = []
    for _ in range(num_points):
        cyl = random.choice(cylinders)
        angle = random.uniform(0, 2 * np.pi)
        height = random.uniform(-cyl['length']/2, cyl['length']/2)
        local_point = np.array([
            cyl['radius'] * np.cos(angle),
            cyl['radius'] * np.sin(angle),
            height
        ])
        global_point = cyl['rotation'] @ local_point + cyl['position']
        points.append(global_point)
    return points

def create_buffered_cylinder(cyl, buffer):
    return {
        'position': cyl['position'],
        'rotation': cyl['rotation'],
        'radius': cyl['radius'] + buffer,
        'length': cyl['length'] + 2 * buffer
    }

def project_to_buffered_cylinder(point, cyl):
    axis = cyl['rotation'] @ np.array([0, 0, 1])
    center = cyl['position']
    rel_point = point - center
    height = np.dot(rel_point, axis)
    height = np.clip(height, -cyl['length']/2, cyl['length']/2)
    radial = rel_point - height * axis
    if np.linalg.norm(radial) > 0:
        radial = cyl['radius'] * radial / np.linalg.norm(radial)
    projected = center + height * axis + radial
    return projected

def find_path_points(inspection_points, cylinders, offset=3):
    path_points = []
    for ip in inspection_points:
        closest = None
        min_dist = float('inf')
        for cyl in cylinders:
            projected = project_to_buffered_cylinder(ip, create_buffered_cylinder(cyl, BUFFER))
            dist = np.linalg.norm(projected - ip)
            if dist < min_dist:
                closest = projected
                min_dist = dist

        # Compute direction vector from inspection to buffer boundary to offset path planning point
        direction = ip - closest
        if np.linalg.norm(direction) > 0:
            direction /= np.linalg.norm(direction)
        offset_point = closest + offset * direction  # final path planning point

        path_points.append(offset_point)
    return path_points

def plot_cylinder(ax, cyl, color='r', alpha=1.0):
    z = np.linspace(-cyl['length']/2, cyl['length']/2, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = cyl['radius'] * np.cos(theta_grid)
    y_grid = cyl['radius'] * np.sin(theta_grid)

    points = np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3)
    rotated = (cyl['rotation'] @ points.T).T + cyl['position']
    X, Y, Z = rotated[:, 0].reshape(z.shape[0], -1), rotated[:, 1].reshape(z.shape[0], -1), rotated[:, 2].reshape(z.shape[0], -1)
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

def visualize_all(cylinders, inspection_points, path_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cyl in cylinders:
        plot_cylinder(ax, cyl, color='red', alpha=1.0)
        plot_cylinder(ax, create_buffered_cylinder(cyl, BUFFER), color='cyan', alpha=0.3)

    insp = np.array(inspection_points)
    path = np.array(path_points)

    ax.scatter(insp[:,0], insp[:,1], insp[:,2], c='red', label='Inspection Points', s=40)
    ax.scatter(path[:,0], path[:,1], path[:,2], c='blue', label='Path Planning Points', s=30)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("model.sdf with Cylinders, Buffered Zones, Inspection & Path Points")
    ax.legend()
    ax.view_init(elev=25, azim=35)
    plt.tight_layout()
    plt.show()
        
def export_path_points_to_csv(path_points, out_path):
    df = pd.DataFrame(path_points, columns=["x", "y", "z"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

# Run on the sdf file
if __name__ == "__main__":
    sdf_path = os.path.join("..", "..", "models", "full_jacket", "model.sdf")
    output_csv = os.path.join("..", "output", "inspection_points.csv")
    cylinders = parse_sdf_with_visual_pose(sdf_path)
    inspection_points = sample_points_on_cylinders(cylinders, NUM_INSPECTION_POINTS)
    path_points = find_path_points(inspection_points, cylinders)
    export_path_points_to_csv(path_points, output_csv)

# Run 3D visualization
    visualize_all(cylinders, inspection_points, path_points)

