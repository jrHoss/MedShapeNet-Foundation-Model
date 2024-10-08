# MedShapeNet Foundation Model

MedShapeNet Foundation Model is a deep learning framework designed for 3D point cloud completion in medical imaging. It leverages a transformer-based architecture and integrates BERT text encoding to reconstruct anatomical structures from incomplete data, trained on the extensive MedShapeNet dataset.

## Dataset

- **MedShapeNet is a large-scale dataset specifically designed for 3D medical shape analysis. It contains over 100,000 3D models of anatomical structures across 240 classes, representing various organs, bones, and other body parts.
- **Our model was trained on 200,000 3D shapes.
- **The MedShapeNet is available [here](https://github.com/GLARKI/MedShapeNet2.0)
![image](https://github.com/user-attachments/assets/eca9600c-d668-4c76-9999-36642c05a595)


## Key Features
- **Transformer-Based Autoencoder** for efficient feature extraction and 3D point cloud completion.
- **Multi-Modal Integration** of text data using BERT to enhance point cloud reconstruction.
- **Density-Aware Chamfer Distance Loss** tailored for handling varying point densities.

## Online Demo
Explore the capabilities of the MedShapeNet Foundation Model with our [online demo](http://gpuserver.di.uminho.pt:36124/).
