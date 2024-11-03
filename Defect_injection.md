
### What This Code Does
This script is designed to inject defects into complete point clouds extracted from 3D mesh files in `.ply` format. Here's a breakdown:

#### Overview
1. **Load Point Clouds**: Traverses a given root directory to find `.ply` files containing complete 3D meshes.
2. **Extract and Normalize**:
   - Loads each mesh using `trimesh`.
   - Extracts vertex data and samples points uniformly.
   - Normalizes the point cloud by centering it at the origin and scaling it to fit within a unit sphere.
3. **Inject Defects**:
   - Selects key points in the point cloud using farthest point sampling (FPS).
   - Removes neighboring points around these key points to create a partial (defective) point cloud.
4. **Prepare Text Labels**: Extracts class names from file names to be used as labels.
5. **Tokenize Text Labels**:
   - Uses a pre-trained BERT tokenizer to convert class names into tokenized input IDs and attention masks.
6. **Output**: Returns arrays of partial point clouds, random eye seeds, tokenized class names, attention masks, and ground truth point clouds.

#### Key Functions
- **`normalize_point_cloud(point_cloud)`**: Centers the point cloud at the origin and scales it to fit within a unit sphere.
- **`remove_knn_points_by_index(points, point_index, num_remove)`**: Removes a specified number of nearest neighbors around a given point index, creating a defect.
- **`preprocess_data(root_folder)`**: Main function to extract, normalize, inject defects, and prepare data for training.

#### Usage
```python
root_folder = "path/to/your/ply/files"  # Directory containing .ply files with complete point clouds
input_set, eye_seeds, input_ids, attention_masks, GT_set = preprocess_data(root_folder)
```
Full code can be found [here](Data/cloud_preprocessing.py)
