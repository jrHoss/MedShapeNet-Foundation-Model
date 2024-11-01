
You can check out the full code for this step [here](MedShapeNet-Foundation-Model/Data/meshes_to_clouds.py)

The URL for the dataset as a .txt can be accessed [here](https://medshapenet.ikim.nrw/uploads/MedShapeNetDataset.txt)

## What This Code Does

This script is designed to extract and process point clouds from 3D mesh files in the MedShapeNet dataset. Here's a breakdown:

### Overview

1. **Fetch Links**: Reads a list of URLs from a text file, each pointing to a 3D mesh file.
2. **Download Meshes**: Fetches each 3D mesh from the provided URLs using `requests`.
3. **Convert to Point Cloud**: 
   - Loads the mesh with `trimesh`.
   - Samples a specified number of points from the surface.
4. **Normalize Point Cloud**: 
   - Centers the point cloud at the origin.
   - Scales it so all points fit within a unit sphere.
5. **Save as PLY File**: Exports the normalized point cloud to a `.ply` file.

### Key Functions

- **normalize_point_cloud(point_cloud)**
  - Centers the point cloud at the origin and scales it to fit within a unit sphere.
- **process_and_save_point_clouds(links_url, point_cloud_dir='', num_points=6144)**
  - Main function to download, convert, normalize, and save point clouds.

### Usage

```python
links_url = "https://example.com/mesh_links.txt"  # URL to text file with 3D mesh links
point_cloud_dir = "output_point_clouds"            # Directory to save point clouds
num_points = 6144                                  # Number of points per cloud

process_and_save_point_clouds(links_url, point_cloud_dir, num_points)
