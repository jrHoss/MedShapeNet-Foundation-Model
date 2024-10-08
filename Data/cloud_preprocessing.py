import os
import numpy as np
import trimesh
import fpsample
from transformers import BertTokenizer

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

def remove_knn_points_by_index(points, point_index, num_remove):
    center_point = points[point_index]
    distances = np.linalg.norm(points - center_point, axis=1)
    knn_indices = np.argsort(distances)[:num_remove]
    remaining_points = np.delete(points, knn_indices, axis=0)
    return remaining_points

def preprocess_data(root_folder):
    """
    Preprocesses point cloud data and text labels for training by generating input sets, eye seeds,
    tokenized text, and ground truth sets from 3D mesh files.

    Args:
        root_folder (str): The root directory containing .ply files with point clouds.

    Returns:
        tuple: A tuple containing six lists:
            - input_set (list): List of partial point clouds generated from the meshes.
            - eye_seeds (list): Randomly generated eye seeds for each point cloud.
            - text_set (list): List of text labels (class names) for each mesh.
            - input_ids (list): List of tokenized text input IDs.
            - attention_masks (list): List of attention masks for the tokenized text.
            - GT_set (list): Ground truth (full) point clouds.
    """
    input_set = []
    eye_seeds = []
    text_set = []
    GT_set = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.ply'):
                base_name = os.path.splitext(filename)[0]
                class_name_part = base_name.split('_', 1)[1]
                class_name = class_name_part.split('.')[0]
                class_name = class_name.replace('_', ' ')
                file_path = os.path.join(dirpath, filename)

                # Load the mesh and sample the ground truth points
                mesh = trimesh.load(file_path)
                GT = np.array(mesh.vertices)
                GT_idx = fpsample.fps_sampling(GT, 6144, start_idx=0)
                GT = GT[GT_idx]

                # Generate partial point clouds using farthest point sampling
                indices = fpsample.fps_sampling(GT, 2, start_idx=0)
                for idx in indices:
                    partial_cloud = remove_knn_points_by_index(GT, idx, 2048)
                    input_set.append(partial_cloud)
                    eye_seeds.append(np.random.rand(1, 1))
                    text_set.append(class_name)
                    GT_set.append(GT)

    # Step 4: Tokenize the text using BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer.batch_encode_plus(
        text_set,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    input_ids = encoded_inputs['input_ids']
    attention_masks = encoded_inputs['attention_mask']

    return np.array(input_set, dtype= np.float32), np.array(eye_seeds, dtype= np.float32), np.array(input_ids, dtype= np.int32), np.array(attention_masks, dtype= np.int32), np.array(GT_set, dtype= np.float32)
