# MedShapeNet Foundation Model

The **MedShapeNet foundation model** is the first multi-modal foundation model for medical **point cloud completion** and serves as a foundation for future research in this area. It is designed to handle incomplete 3D point cloud data and reconstruct the full shape of various medical structures. By combining both 3D geometric data and textual data, this model enhances accuracy in shape reconstruction, supporting more precise analysis and potential applications in extended reality (XR) for medicine and custom bone implant design.


## Dataset: [MedShapeNet](https://github.com/GLARKI/MedShapeNet2.0).

The model is trained on the **MedShapeNet dataset**, a comprehensive collection of over **100,000 3D medical shapes**. This dataset encompasses a wide range of medical structures, including **organs, vessels, bones, instruments**, and more, spanning across **240 distinct classes**.

To create a robust training set for our model:
- We extracted **point clouds** from the vertices of each 3D mesh file in MedShapeNet.
- To simulate real-world scenarios where data might be incomplete, we introduced **defects by removing points** from each point cloud. This created an "incomplete" input that the model aims to reconstruct.
- Each point cloud was processed twice in this way, generating a total of **200,000 point clouds** for training.

## Multi-Modal Approach

To enhance the model's interpretative ability, we provided **class names as textual input**. This allows the model to differentiate between classes, such as distinguishing a **healthy liver** from a **tumorous liver**, adding a layer of semantic understanding to its point cloud completion.

The MedShapeNet foundation model demonstrates the potential of multi-modal learning in medical applications, bridging 3D shape data with textual descriptors to improve the quality and accuracy of shape completion in medical imaging.












# MedShapeNet Foundation Model

MedShapeNet Foundation Model is a deep learning framework designed for 3D point cloud completion in medical imaging. It leverages a transformer-based architecture and integrates BERT text encoding to reconstruct anatomical structures from incomplete data, trained on the extensive MedShapeNet dataset.

## Dataset

- MedShapeNet is a large-scale dataset specifically designed for 3D medical shape analysis. It contains over 100,000 3D models of anatomical structures across 240 classes, representing various organs, bones, and other body parts.
- Our model was trained on 200,000 3D shapes.
- The MedShapeNet is available [here](https://github.com/GLARKI/MedShapeNet2.0).

![MSN_meshes_examples2](https://github.com/user-attachments/assets/6a66b8e2-71cd-4bcf-afa3-eef3ffbdc720)

## Key Features
- Transformer-Based Autoencoder** for efficient feature extraction and 3D point cloud completion.
- Multi-Modal Integration** of text data using BERT to enhance point cloud reconstruction.
- Density-Aware Chamfer Distance Loss** tailored for handling varying point densities.

## Model weights
the model weights are available [here](https://uni-duisburg-essen.sciebo.de/s/j459KveLeZ98qBc/download).

## Paper
The paper can be accessed [here](MedShapeNet_Foundation_Model.pdf), and on [ResearchGate](https://www.researchgate.net/publication/384968432_A_MedShapeNet_Foundation_Model_-_Learning-Based_Multimodal_Medical_Point_Cloud_Completion)


<!--## Online Demo
<!--Explore the capabilities of the MedShapeNet Foundation Model with our [online demo](http://gpuserver.di.uminho.pt:36124/).


![imgpsh_fullsize_anim (1)](https://github.com/user-attachments/assets/d25d1eb5-7f78-4e55-bb4b-f6ab00a0957d)
