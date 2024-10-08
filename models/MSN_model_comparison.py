from tensorflow.keras import Input
from tensorflow.keras import models as M
from tensorflow.keras import layers as L
from utils.MSN_utils import *

def PCT_encoder(xyz):
    x = LBR(xyz, 64, 'E-IN_LBR1', use_bias=False)
    x = LBR(x, 128, 'E-IN_LBR2', use_bias=False)
    fps_idx = UniformSampler(4096)(xyz)
    new_xyz, new_feature = L.Lambda(sample_and_group, arguments={'nsample':32}, name='E-SG1')([xyz, x, fps_idx])
    x = LBR(new_feature, 512, 'E-SG1_LBR1', use_bias=False)
    x = L.Lambda(lambda t: tf.reduce_max(t, axis=2), name='E-SG1_MaxPool')(x)
    fps_idx = UniformSampler(2048)(new_xyz)
    new_xyz, new_feature = L.Lambda(sample_and_group, arguments={'nsample':32}, name='E-SG2')([new_xyz, x, fps_idx])
    x = LBR(new_feature, 1024, 'E-SG2_LBR1', use_bias=False)
    x = L.Lambda(lambda t: tf.reduce_max(t, axis=2), name='E-SG2_MaxPool')(x)
    x1 = SelfAttention(x, 'E-SA1')
    x2 = SelfAttention(x1, 'E-SA2')
    x3 = SelfAttention(x2, 'E-SA3')
    x4 = SelfAttention(x3, 'E-SA4')
    x0 = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='E-SA_Concat')([x1,x2,x3,x4])
    x = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='E-OUT_Concat')([x0,x])
    x = LBR(x, 2048, 'E-OUT_LBR', use_bias=False, LeakyAlpha=0.2)
    x1 = SelfAttention(x, 'E-SA5')
    x2 = SelfAttention(x1, 'E-SA6')
    x3 = SelfAttention(x2, 'E-SA7')
    x4 = SelfAttention(x3, 'E-SA8')
    x0 = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='E-SA_Concat2')([x1,x2,x3,x4])
    x = LBR(x0, 4096, 'E-OUT_LBR1', use_bias=False, LeakyAlpha=0.2)
    output_feats = L.Lambda(lambda t: tf.reduce_max(t, axis=1, keepdims=True), name='E-OUT_MaxPool')(x)
    return output_feats

def pct_decoder(input_feats, input_eye_seed):
    m_feats = L.Lambda(lambda x: tf.tile(x, [1,1024,1]), name = 'D-IN_replicate')(input_feats)
    input_eye = input_eye_seed + tf.eye(1024,1024)
    x = L.Dense(4096//4, use_bias=False, activation=None, name='D1-IN')(input_eye)
    x1 = SourceTargetAttention([m_feats,x] , 'D-STA1')
    x2 = SourceTargetAttention([m_feats,x1], 'D-STA2')
    x3 = SourceTargetAttention([m_feats,x2], 'D-STA3')
    x4 = SourceTargetAttention([m_feats,x3], 'D-STA4')
    x0 = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='D1-STA_Concat')([x1,x2,x3,x4])
    x = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='D1-OUT_Concat')([x0,x])
    m_feats2 = Upsampling(x, 3, 'D1-OUT_CopyAndMapping')
    input_eye2 = input_eye_seed + tf.eye(3072,3072)
    x = L.Dense(1024//4, use_bias=False, activation=None, name='D2-IN')(input_eye2)
    x1 = SourceTargetAttention([m_feats2,x] , 'D2-STA1')
    x2 = SourceTargetAttention([m_feats2,x1], 'D2-STA2')
    x3 = SourceTargetAttention([m_feats2,x2], 'D2-STA3')
    x4 = SourceTargetAttention([m_feats2,x3], 'D2-STA4')
    x0 = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='D2-STA_Concat')([x1,x2,x3,x4])
    x = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='D2-OUT_Concat')([x0,x])
    x = Upsampling(x, 2, 'D2-OUT_CopyAndMapping')
    x = LBR(x,128, 'D-OUT_LBR1', use_bias=False)
    x = LBR(x,128, 'D-OUT_LBR2', use_bias=False)
    x = LBR(x, 64, 'D-OUT_LBR3', use_bias=False, LeakyAlpha=0.2)
    output_points = L.Dense(3, activation=None, name='D-OUT_lin')(x)
    return output_points



def AutoEncoder(num_input_points = 4096):
  eye_seed = Input(shape=(1, 1), name='input_eye_seed')
  xyz = Input(shape=(num_input_points, 3), name='input_points')
  encoded = PCT_encoder(xyz)
  output = pct_decoder(encoded, eye_seed)
  return M.Model(inputs = [xyz, eye_seed], outputs = output)
