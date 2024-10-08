def eye_seed(X):
    return tf.zeros([X.shape[0],1,1])

def pairwise_distance(xyz1, xyz2):
    n = xyz1.shape[1]
    c = xyz1.shape[2]
    m = xyz2.shape[1]
    xyz1 = tf.tile(tf.reshape(xyz1, (-1,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (-1,m,1,c)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    return dist

def knn_point(k, xyz1, xyz2):
    dist = -pairwise_distance(xyz1, xyz2)
    val, idx = tf.math.top_k(dist, k)
    return -val, idx

class UniformSampler(tf.keras.layers.Layer):
    def __init__(self, num_points, seed=42, **kwargs):
        super(UniformSampler, self).__init__(**kwargs)
        self.num_points = num_points
        self.seed = seed

    def build(self, input_shape):
        pass

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        data_size = tf.shape(inputs)[1]
        indices = tf.random.uniform(
            shape=(batch_size, self.num_points),
            minval=0,
            maxval=data_size,
            dtype=tf.int32,
            seed=self.seed
        )
        return indices

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_points, input_shape[2])

    def get_config(self):
        config = super(UniformSampler, self).get_config()
        config.update({
            "num_points": self.num_points,
            "seed": self.seed
        })
        return config

def sample_and_group(args, nsample):
    xyz, pts, fps_idx = args
    new_xyz = tf.gather_nd(xyz, tf.expand_dims(fps_idx,-1), batch_dims=1)
    new_pts = tf.gather_nd(pts, tf.expand_dims(fps_idx,-1), batch_dims=1)
    _, idx = knn_point(nsample, xyz, new_xyz)
    grouped_pts = tf.gather_nd(pts, tf.expand_dims(idx,-1), batch_dims=1)
    grouped_pts -= tf.tile(tf.expand_dims(new_pts, 2),
                           (1,1,nsample,1))
    new_pts = tf.concat([grouped_pts,
                         tf.tile(tf.expand_dims(new_pts, 2),
                                 (1,1,nsample,1))],
                        axis=-1)
    return new_xyz, new_pts

def LBR(tensor, C, seq_name, use_bias=True, activation=None, LeakyAlpha=0.0):
    x_in = Input(shape=tensor.shape[1:], name=seq_name+'_input')
    x = L.Dense(C, use_bias=use_bias, activation=activation, name=seq_name+'_lin')(x_in)
    if LeakyAlpha==0.0:
        x_out = L.ReLU(name=seq_name+'_ReLU')(x)
    else:
        x_out = L.LeakyReLU(alpha=LeakyAlpha, name=seq_name+'_ReLU')(x)
    model = M.Model(inputs=x_in, outputs=x_out, name=seq_name)
    return model(tensor)

def SelfAttention(tensor, seq_name):
    x_in = Input(shape=tensor.shape[1:], name=seq_name+'_input')
    C = x_in.shape[2]
    W_q = L.Dense(C//4, use_bias=False, activation=None, name=seq_name+'_Q')
    W_k = L.Dense(C//4, use_bias=False, activation=None, name=seq_name+'_K')
    W_v = L.Dense(C, use_bias=False, activation=None, name=seq_name+'_V')
    x_q = W_q(x_in)
    x_k = W_k(x_in)
    W_k.set_weights(W_q.get_weights())
    x_k = L.Lambda(lambda t: tf.transpose(t, perm=(0,2,1)), name=seq_name+'_KT')(x_k)
    x_v = W_v(x_in)
    energy = L.Lambda(lambda ts: tf.matmul(ts[0],ts[1]), name=seq_name+'_matmul1')([x_q, x_k])
    attention = L.Softmax(axis=1, name=seq_name+'_softmax')(energy)
    attention = L.Lambda(lambda t: t / (1e-9 + tf.reduce_sum(t, axis=2, keepdims=True)), name=seq_name+'_l1norm')(attention)
    x_r = L.Lambda(lambda ts: tf.matmul(ts[0],ts[1]), name=seq_name+'_matmul2')([attention, x_v])
    x_r = L.Lambda(lambda ts: tf.subtract(ts[0],ts[1]), name=seq_name+'_subtract')([x_in, x_r])
    x_r = LBR(x_r, C, seq_name+'_LBR', use_bias=True)
    x_out = L.Lambda(lambda ts: tf.add(ts[0],ts[1]), name=seq_name+'_add')([x_in, x_r])
    model = M.Model(inputs=x_in, outputs=x_out, name=seq_name)
    return model(tensor)


def SourceTargetAttention(args, seq_name):
    E_tensor, D_tensor = args
    xE_in = Input(shape=E_tensor.shape[1:], name=seq_name+'_input-E')
    C = xE_in.shape[2]
    xD_in = Input(shape=D_tensor.shape[1:], name=seq_name+'_input-D')
    out_dim = xD_in.shape[2]
    W_q = L.Dense(C//4, use_bias=False, activation=None, name=seq_name+'_Q')
    W_k = L.Dense(C//4, use_bias=False, activation=None, name=seq_name+'_K')
    W_v = L.Dense(out_dim, use_bias=False, activation=None, name=seq_name+'_V')
    x_q = W_q(xD_in)
    x_k = W_k(xE_in)
    x_k = L.Lambda(lambda t: tf.transpose(t, perm=(0,2,1)), name=seq_name+'_KT')(x_k)
    x_v = W_v(xE_in)
    energy = L.Lambda(lambda ts: tf.matmul(ts[0],ts[1]), name=seq_name+'_matmul1')([x_q, x_k])
    attention = L.Softmax(axis=1, name=seq_name+'_softmax')(energy)
    attention = L.Lambda(lambda t: t / (1e-9 + tf.reduce_sum(t, axis=2, keepdims=True)), name=seq_name+'_l1norm')(attention)
    x_r = L.Lambda(lambda ts: tf.matmul(ts[0],ts[1]), name=seq_name+'_matmul2')([attention, x_v])
    x_r = L.Lambda(lambda ts: tf.subtract(ts[0],ts[1]), name=seq_name+'_subtract')([xD_in, x_r])
    x_r = LBR(x_r, out_dim, seq_name+'_LBR', use_bias=True)
    x_out = L.Lambda(lambda ts: tf.add(ts[0],ts[1]), name=seq_name+'_add')([xD_in, x_r])
    model = M.Model(inputs=[xE_in,xD_in], outputs=x_out, name=seq_name)
    return model([E_tensor,D_tensor])

def Upsampling(tensor, nmul, seq_name):
    x_in = Input(shape=tensor.shape[1:], name=seq_name+'_input')
    x = L.Lambda(lambda t: tf.expand_dims(t, 2), name=seq_name+'_expand')(x_in)
    C = x.shape[-1]//nmul
    x1 = L.Conv2DTranspose(C,(1,nmul),(1,nmul), use_bias=True, activation=None, name=seq_name+'_convT')(x)
    x2 = L.Dense(C, use_bias=True, activation=None, name=seq_name+'_lin')(x)
    x2 = L.Lambda(lambda t: tf.tile(t, [1,1,nmul,1]), name=seq_name+'_tile')(x2)
    x = L.Lambda(lambda ts: tf.add(ts[0],ts[1]), name=seq_name+'_add')([x1, x2])
    npoint = x.shape[1]*x.shape[2]
    x_out = L.Lambda(lambda t: tf.reshape(t, [-1,npoint,t.shape[3]]), name=seq_name+'_reshape')(x)
    model = M.Model(inputs=x_in, outputs=x_out, name=seq_name)
    return model(tensor)

