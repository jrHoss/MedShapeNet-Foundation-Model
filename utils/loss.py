import tensorflow as tf
import tensorflow_graphics as tfg
from tensorflow_graphics.nn.loss import chamfer_distance

def distance_matrix(array1, array2):
    batch_size, num_point, num_features = array1.shape
    expanded_array1 = tf.tile(tf.expand_dims(array1, 2), [1, 1, num_point, 1])
    expanded_array2 = tf.tile(tf.expand_dims(array2, 1), [1, num_point, 1, 1])
    distances = tf.norm(expanded_array1-expanded_array2, axis=-1)
    return distances

def min_distances_and_indices(array1, array2):
    distances = distance_matrix(array1, array2)
    min_dists_1_to_2, indices_1_to_2 = tf.reduce_min(distances, axis=-1), tf.argmin(distances, axis=-1)
    min_dists_2_to_1, indices_2_to_1 = tf.reduce_min(distances, axis=-2), tf.argmin(distances, axis=-2)
    return min_dists_1_to_2, min_dists_2_to_1, indices_1_to_2, indices_2_to_1

def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    dist1, dist2, idx1, idx2 = min_distances_and_indices(gt, output)
    cd_p = (tf.sqrt(tf.reduce_mean(dist1, axis=1)) + tf.sqrt(tf.reduce_mean(dist2, axis=1))) / 2
    cd_t = (tf.reduce_mean(dist1, axis=1) + tf.reduce_mean(dist2, axis=1))
    if separate:
        res = [tf.concat([tf.reduce_mean(tf.sqrt(dist1), axis=1, keepdims=True),
                          tf.reduce_mean(tf.sqrt(dist2), axis=1, keepdims=True)], axis=0),
               tf.concat([tf.reduce_mean(dist1, axis=1, keepdims=True),
                          tf.reduce_mean(dist2, axis=1, keepdims=True)], axis=0)]
    else:
        res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2, 0.0001)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res

def calc_dcd(x, gt, alpha=1, n_lambda=1, return_raw=False, non_reg=False):
    x = tf.cast(x, tf.float32)
    gt = tf.cast(gt, tf.float32)
    batch_size = tf.shape(x)[0]
    n_x = tf.shape(x)[1]
    n_gt = tf.shape(gt)[1]
    if non_reg:
        frac_12 = tf.maximum(1.0, tf.cast(n_x, tf.float32) / tf.cast(n_gt, tf.float32))
        frac_21 = tf.maximum(1.0, tf.cast(n_gt, tf.float32) / tf.cast(n_x, tf.float32))
    else:
        frac_12 = tf.cast(n_x, tf.float32) / tf.cast(n_gt, tf.float32)
        frac_21 = tf.cast(n_gt, tf.float32) / tf.cast(n_x, tf.float32)
    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    exp_dist1 = tf.exp(-dist1 * alpha)
    exp_dist2 = tf.exp(-dist2 * alpha)
    def compute_loss(b):
        idx1_b = tf.gather(idx1, b)
        idx2_b = tf.gather(idx2, b)
        count1 = tf.math.bincount(idx1_b, minlength=tf.cast(n_x, tf.int64))
        weight1 = tf.gather(count1, idx1_b)
        weight1 = tf.cast(weight1, tf.float32)
        weight1 = tf.pow(weight1, n_lambda)
        weight1 = tf.pow((weight1 + 1e-6), -1) * frac_21
        loss1 = tf.reduce_mean(-exp_dist1[b] * weight1 + 1.0)
        count2 = tf.math.bincount(idx2_b, minlength=tf.cast(n_gt, tf.int64))
        weight2 = tf.gather(count2, idx2_b)
        weight2 = tf.cast(weight2, tf.float32)
        weight2 = tf.pow(weight2, n_lambda)
        weight2 = tf.pow((weight2 + 1e-6), -1) * frac_12
        loss2 = tf.reduce_mean(-exp_dist2[b] * weight2 + 1.0)
        return loss1, loss2
    loss1, loss2 = tf.map_fn(compute_loss, tf.range(batch_size), dtype=(tf.float32, tf.float32))
    loss = tf.reduce_mean(loss1 + loss2)
    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return loss

def chamfer_distance_loss(y_true, y_pred):
    return tfg.nn.loss.chamfer_distance.evaluate(y_true, y_pred)
