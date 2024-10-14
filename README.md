# MedShapeNet Foundation Model

MedShapeNet Foundation Model is a deep learning framework designed for 3D point cloud completion in medical imaging. It leverages a transformer-based architecture and integrates BERT text encoding to reconstruct anatomical structures from incomplete data, trained on the extensive MedShapeNet dataset.

## Dataset

- MedShapeNet is a large-scale dataset specifically designed for 3D medical shape analysis. It contains over 100,000 3D models of anatomical structures across 240 classes, representing various organs, bones, and other body parts.
- Our model was trained on 200,000 3D shapes.
- The MedShapeNet is available [here](https://github.com/GLARKI/MedShapeNet2.0).

![imgpsh_fullsize_anim (1)](https://github.com/user-attachments/assets/d25d1eb5-7f78-4e55-bb4b-f6ab00a0957d)

## Key Features
- Transformer-Based Autoencoder** for efficient feature extraction and 3D point cloud completion.
- Multi-Modal Integration** of text data using BERT to enhance point cloud reconstruction.
- Density-Aware Chamfer Distance Loss** tailored for handling varying point densities.

%## Online Demo
%Explore the capabilities of the MedShapeNet Foundation Model with our [online demo](http://gpuserver.di.uminho.pt:36124/).

## Model weights
the model weights are available [here](https://uni-duisburg-essen.sciebo.de/s/j459KveLeZ98qBc/download).
