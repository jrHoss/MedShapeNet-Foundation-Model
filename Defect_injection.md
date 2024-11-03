### What This Code Does
This script is designed to inject defects into complete point clouds extracted from 3D mesh files in `.ply` format. Here's a breakdown:

#### Overview
1. **Load Point Clouds**: Traverses a given root directory to find `.ply` files containing complete 3D meshes.
2. **Extract and Normalize**:
   - Loads each mesh using `trimesh`.
   - Extracts vertex data and samples points uniformly.
   - Normalizes the point cloud by centering it at the origin and scaling it to fit within a unit sphere.
3. **Inject Defects**:
   - Uses **farthest point sampling (FPS)** to select key points in the point cloud. This ensures a diverse and evenly distributed selection of points to create defects.
   - Removes neighboring points around these key points using **K-Nearest Neighbors (KNN)**. The `remove_knn_points_by_index` function identifies and removes a specified number of nearest points, simulating a defect in the point cloud.

![image](https://github.com/user-attachments/assets/e34c795c-2b48-4c7c-824d-90e6161e454f)

4. **Prepare Text Labels**: Extracts class names from file names to be used as labels.
5. **Tokenize Text Labels**:
   - Uses a pre-trained BERT tokenizer to convert class names into tokenized input IDs and attention masks.
6. **Output**: Returns arrays of partial point clouds, random eye seeds, tokenized class names, attention masks, and ground truth point clouds.

#### Key Functions
- **`normalize_point_cloud(point_cloud)`**: Centers the point cloud at the origin and scales it to fit within a unit sphere.
- **`remove_knn_points_by_index(points, point_index, num_remove)`**: Uses K-Nearest Neighbors to remove a specified number of points around a given index, creating a defect in the point cloud.
- **`preprocess_data(root_folder)`**: Main function to extract, normalize, use farthest point sampling to select points, inject defects using KNN, and prepare data for training.

#### Usage
```python
root_folder = "path/to/your/ply/files"  # Directory containing .ply files with complete point clouds
input_set, eye_seeds, input_ids, attention_masks, GT_set = preprocess_data(root_folder)
```

