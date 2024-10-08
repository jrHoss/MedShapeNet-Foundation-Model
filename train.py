# Required Imports
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from Data.mesh_to_point_cloud import process_and_save_point_clouds
from Data.data_preprocessing import preprocess_data
from models.MSN_foundation_model import PCT_AE_Multimodal  # Adjust the import to where your model class is located
from utils.loss import calc_dcd  # Ensure you have your custom loss function in the utils directory
from transformers import BertTokenizer

# URL pointing to the MedShapeNet dataset file containing links to 3D mesh files
MSN_url = "https://medshapenet.ikim.nrw/uploads/MedShapeNetDataset.txt"

# Directory to save the processed point cloud data
Data_directory = "point_clouds"

# Step 1: Download the mesh files and convert them to point clouds
process_and_save_point_clouds(MSN_url, point_cloud_dir=Data_directory, num_points=6144)

# Step 2: Preprocess the data by loading the generated point clouds and text labels for model training
input_set, eye_seeds, text_set, input_ids, attention_masks, GT_set = preprocess_data(root_folder='path/to/ply/files')

# Step 3: Define a learning rate scheduler to reduce the learning rate when the validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',        # Monitor the validation loss during training
    factor=0.9,                # Factor by which the learning rate will be reduced
    patience=5,                # Number of epochs with no improvement before the learning rate is reduced
    verbose=1,                 # Verbosity mode (1 = progress messages)
    mode='auto',               # Mode for deciding when to reduce learning rate ('min' or 'max')
    min_lr=1e-20               # Minimum learning rate limit
)

# Step 4: Define a checkpoint callback to save the best model weights during training
save_dir = 'MSN_checkpoint'  # Directory to save the checkpoints
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(save_dir, 'MSN_weights2.h5'),  # Path to save the best weights
    monitor='val_loss',        # Monitor the validation loss to decide when to save the model
    verbose=1,                 # Verbosity mode (1 = progress messages)
    save_best_only=True,       # Save only the best weights based on the monitored metric
    save_weights_only=True,    # Save only the weights (not the entire model)
    mode='min'                 # Mode for saving ('min' because we want to minimize validation loss)
)

# Step 5: Initialize the PCT_AE_Multimodal model with BERT and point cloud components
AE = PCT_AE_Multimodal(bert_model=bert_model, PCT_encoder=PCT_encoder, pct_decoder=pct_decoder)
AE = AE.model  # Retrieve the underlying Keras model object

# Step 6: Compile the model with an optimizer and the custom loss function
initial_learning_rate = 1e-7  # Set the initial learning rate for the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
AE.compile(optimizer=optimizer, loss=calc_dcd)  # Compile the model with the Adam optimizer and custom loss

# Step 7: Train the model on the prepared data with a validation split and callbacks for optimization
AE.fit(
    x=[input_set, eye_seeds, input_ids, attention_masks],  # Input data for training
    y=GT_set,                                             # Ground truth labels (point clouds)
    epochs=500,                                           # Number of training epochs
    shuffle=True,                                         # Shuffle the data at the start of each epoch
    validation_split=0.1,                                 # Fraction of the data to use as validation
    batch_size=8,                                         # Batch size for training
    verbose=1,                                            # Verbosity mode (1 = progress messages)
    callbacks=[reduce_lr, checkpoint_callback]            # List of callbacks to use during training
)
