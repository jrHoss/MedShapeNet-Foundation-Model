# MedShapeNet Foundation Model


<div align="center">
  <img src="https://github.com/user-attachments/assets/0bc5063e-5512-4f11-afc5-128b8eacd02f" alt="MSN_body" width="200" height="350"/>
</div>




<div align="center">
 <img src="https://github.com/user-attachments/assets/abf6dc67-0353-49ba-b8f3-d796c628106c" alt="MSN FM Logo" width="200" height="400"/>
</div>



<div align="center">
  <img src="https://github.com/user-attachments/assets/3394fcd2-4756-43c9-9f78-3154afbb18eb" alt="Recording 2024-11-22 093020">
</div>




The **MedShapeNet foundation model** is the first multi-modal foundation model for medical **point cloud completion** and serves as a foundation for future research in this area. It is designed to handle incomplete 3D point cloud data and reconstruct the full shape of various medical structures. By combining both 3D point cloud data and textual data, this model enhances accuracy in shape reconstruction, supporting more precise analysis and potential applications in extended reality (XR) for medicine and custom bone implant design.

## Key Features
- Transformer-Based Autoencoder** for efficient feature extraction and 3D point cloud completion.
- Multi-Modal Integration** of text data using BERT to enhance point cloud reconstruction.
- Density-Aware Chamfer Distance Loss** tailored for handling varying point densities.


## Dataset: [MedShapeNet](https://github.com/GLARKI/MedShapeNet2.0).

The model is trained on the **MedShapeNet dataset**, a comprehensive collection of over **100,000 3D medical shapes**. This dataset encompasses a wide range of medical structures, including **organs, vessels, bones, instruments**, and more, spanning across **240 distinct classes**.



![Presentation1-removebg](https://github.com/user-attachments/assets/1e2be539-f23e-42d0-b73a-ac274f97a19c)





To create a robust training set for our model:
- We extracted **point clouds** from the vertices of each 3D mesh file in MedShapeNet.
- Point extactiom is explained in detail [here](Point_cloud_extraction.md)
- To simulate real-world scenarios where data might be incomplete, we introduced **defects by removing points** from each point cloud. This created an "incomplete" input that the model aims to reconstruct.
- Each point cloud was processed twice in this way, generating a total of **200,000 point clouds** for our dataset. 90% was designated for training and 10% for validation.
- Defect injection and augmentation are explained [here](Defect_injection.md)



<!-- ![image](https://github.com/user-attachments/assets/e34c795c-2b48-4c7c-824d-90e6161e454f) -->


## Multi-Modal Approach

To enhance the model's interpretative ability, we provided **class names as textual input**. This allows the model to differentiate between classes, such as distinguishing a **healthy liver** from a **tumorous liver**, adding a layer of semantic understanding to its point cloud completion.

The MedShapeNet foundation model demonstrates the potential of multi-modal learning in medical applications, bridging 3D shape data with textual descriptors to improve the quality and accuracy of shape completion in medical imaging.




<div align="center">
  <img src="https://github.com/user-attachments/assets/7a1ce76a-8065-45c7-88d8-175f9cfc9e4a" alt="image">
</div>


## Demo Point cloud dataset
- A small demo [dataset](demo_point_clouds.zip) of 220 point clouds have been preprocessed and can be used directly with the model for inference.

## Training Demo

- This a [Notebook](MSN_model_training_Demo.ipynb) on how to build and train the model.

## Inference Demo

- This a [Notebook](MSN_model_inference_demo.ipynb) on how to use the model to run inference and visualize the results.


## Model weights
the model weights are available [here](https://uni-duisburg-essen.sciebo.de/s/j459KveLeZ98qBc/download).

## Paper
The paper can be accessed [here](MedShapeNet_Foundation_Model.pdf), and on [ResearchGate](https://www.researchgate.net/publication/384968432_A_MedShapeNet_Foundation_Model_-_Learning-Based_Multimodal_Medical_Point_Cloud_Completion)


<!--## Online Demo
<!--Explore the capabilities of the MedShapeNet Foundation Model with our [online demo](http://gpuserver.di.uminho.pt:36124/).


![imgpsh_fullsize_anim (1)](https://github.com/user-attachments/assets/d25d1eb5-7f78-4e55-bb4b-f6ab00a0957d)
