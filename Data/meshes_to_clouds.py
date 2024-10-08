import os
import requests
import numpy as np
import trimesh
from io import BytesIO
from urllib.parse import urlparse, parse_qs


def normalize_point_cloud(point_cloud):
    # Compute the mean of the point cloud
    mean = np.mean(point_cloud, axis=0)

    # Subtract the mean to move the point cloud to the origin
    point_cloud -= mean

    # Compute the maximum distance from the origin
    max_distance = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))

    # Scale the distances to the range -1.0 and 1.0
    point_cloud /= max_distance

    return point_cloud


def process_and_save_point_clouds(links_url, point_cloud_dir='', num_points=6144):
    """
    Fetches 3D mesh files from links provided in a text file, converts them to point clouds,
    and saves the point clouds as PLY files.
    
    Args:
        links_url (str): URL of the text file containing links to the 3D mesh files.
        point_cloud_dir (str): Directory to save the point cloud PLY files.
        num_points (int): Number of points to sample from each mesh to create the point cloud.
    """
    # Step 1: Read the file containing the links
    def read_links(url):
        response = requests.get(url)
        response.raise_for_status()
        links = response.text.splitlines()
        return links

    links = read_links(links_url)
    
    # Ensure the directory exists
    os.makedirs(point_cloud_dir, exist_ok=True)
    
    # Step 2: Convert the 3D meshes to point clouds directly from URLs
    def mesh_to_point_cloud_from_url(url, num_points):
        response = requests.get(url)
        response.raise_for_status()
        mesh = trimesh.load(BytesIO(response.content), file_type='stl')
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points

    # Step 3: Save the point cloud as a PLY file
    def save_point_cloud_as_ply(points, save_path):
        point_cloud = trimesh.PointCloud(points)
        point_cloud.export(save_path)

    
    #def save_point_cloud_as_txt(points, save_path):
        #np.savetxt(save_path, points, delimiter=' ')

    # Extract the file name from the URL
    def extract_file_name(url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        file_name = query_params.get('files', [None])[0]
        return file_name

    # Process each link, convert to point cloud and save as a PLY file
    for url in links:
        try:
            file_name = extract_file_name(url)
            if file_name is None:
                print(f"Failed to extract file name from {url}")
                continue
            
            point_cloud = mesh_to_point_cloud_from_url(url, num_points)
            base_name = file_name.replace('.stl', '.ply')
            point_cloud_path = os.path.join(point_cloud_dir, base_name)
            save_point_cloud_as_ply(normalize_point_cloud(point_cloud), point_cloud_path)
            #save_point_cloud_as_txt(normalize_point_cloud(point_cloud), point_cloud_path)
            #print(f"Saved point cloud to {point_cloud_path}")
        except Exception as e:
            print(f"Failed to process {url}: {e}")
