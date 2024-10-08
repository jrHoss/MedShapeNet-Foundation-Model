import tensorflow as tf

class PointCloudCompletionModel(tf.keras.Model):
    def __init__(self, alpha=0.5):
        super(PointCloudCompletionModel, self).__init__()
        self.num_coarse = 676  # Adjusted to produce 6084 output points (676 * 9 = 6084)
        self.grid_size = 3
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # Should be 6084
        self.alpha = alpha

        # Encoder layers
        self.encoder_conv1 = tf.keras.layers.Conv1D(128, 1, activation='relu')
        self.encoder_conv2 = tf.keras.layers.Conv1D(256, 1, activation='relu')
        self.encoder_conv3 = tf.keras.layers.Conv1D(512, 1, activation='relu')
        self.encoder_conv4 = tf.keras.layers.Conv1D(1024, 1, activation='relu')

        # Decoder layers
        self.decoder_dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.decoder_dense2 = tf.keras.layers.Dense(1024, activation='relu')
        self.decoder_dense3 = tf.keras.layers.Dense(self.num_coarse * 3)

        self.folding_conv1 = tf.keras.layers.Conv1D(512, 1, activation='relu')
        self.folding_conv2 = tf.keras.layers.Conv1D(512, 1, activation='relu')
        self.folding_conv3 = tf.keras.layers.Conv1D(3, 1)

    def call(self, inputs):
        features = self.create_encoder(inputs)
        coarse, fine = self.create_decoder(features)
        return {'coarse_output': coarse, 'fine_output': fine}

    def create_encoder(self, inputs):
        # inputs: [batch_size, num_points, 3]
        x = self.encoder_conv1(inputs)
        x = self.encoder_conv2(x)
        features_global = tf.reduce_max(x, axis=1, keepdims=True)  # [batch_size, 1, 256]
        features_global = tf.tile(features_global, [1, tf.shape(inputs)[1], 1])  # [batch_size, num_points, 256]
        x = tf.concat([x, features_global], axis=-1)  # [batch_size, num_points, 512]
        x = self.encoder_conv3(x)
        x = self.encoder_conv4(x)
        features = tf.reduce_max(x, axis=1)  # [batch_size, 1024]
        return features

    def create_decoder(self, features):
        x = self.decoder_dense1(features)
        x = self.decoder_dense2(x)
        x = self.decoder_dense3(x)  # [batch_size, num_coarse * 3]
        coarse = tf.reshape(x, [-1, self.num_coarse, 3])  # [batch_size, num_coarse, 3]

        batch_size = tf.shape(features)[0]

        # Generate grid
        grid_x, grid_y = tf.meshgrid(
            tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size),
            tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
        )
        grid = tf.stack([grid_x, grid_y], axis=-1)  # [grid_size, grid_size, 2]
        grid = tf.reshape(grid, [1, -1, 2])  # [1, grid_size^2, 2]
        grid_feat = tf.tile(grid, [batch_size, self.num_coarse, 1])  # [batch_size, num_coarse * grid_size^2, 2]

        # Repeat coarse points
        point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = tf.reshape(point_feat, [batch_size, -1, 3])  # [batch_size, num_coarse * grid_size^2, 3]

        # Global features
        global_feat = tf.tile(tf.expand_dims(features, 1), [1, tf.shape(point_feat)[1], 1])  # [batch_size, num_fine, 1024]

        # Concatenate features
        feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)  # [batch_size, num_fine, ...]

        # Folding layers
        x = self.folding_conv1(feat)
        x = self.folding_conv2(x)
        fine = self.folding_conv3(x) + point_feat  # [batch_size, num_fine, 3]

        return coarse, fine




