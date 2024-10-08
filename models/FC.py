
class FC(tf.keras.Model):
def __init__(self, num_output_points=16384):
    super(Model, self).__init__()
    self.num_output_points = num_output_points
    self.encoder_0 = self.create_encoder_0()
    self.encoder_1 = self.create_encoder_1()
    self.decoder = self.create_decoder()

def create_encoder_0(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(128, 1, activation='relu'),
        tf.keras.layers.Conv1D(256, 1, activation='relu')
    ])
    return model

def create_encoder_1(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(512, 1, activation='relu'),
        tf.keras.layers.Conv1D(1024, 1, activation='relu')
    ])
    return model

def create_decoder(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(self.num_output_points * 3)
    ])
    return model

def call(self, inputs):
    # Encoder Part
    features = self.encoder_0(inputs)
    features_global = tf.reduce_max(features, axis=1, keepdims=True)
    features_concat = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
    features_final = self.encoder_1(features_concat)
    features_max = tf.reduce_max(features_final, axis=1)

    # Decoder Part
    outputs = self.decoder(features_max)
    outputs = tf.reshape(outputs, [-1, self.num_output_points, 3])
    return outputs


