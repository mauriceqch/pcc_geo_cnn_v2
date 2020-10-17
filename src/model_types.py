import logging
import pprint
from enum import Enum

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
from scipy.spatial import cKDTree

from model_opt import compute_optimal_thresholds
from model_transforms import TransformType
from utils.focal_loss import focal_loss
from utils.octree_coding import departition_octree
from utils.patch_gaussian_conditional import patch_gaussian_conditional
from utils.pc_metric import compute_metrics

logger = logging.getLogger(__name__)

# Patch GaussianConditional to obtain debug information
tfc.GaussianConditional = patch_gaussian_conditional(tfc.GaussianConditional)


def pc_to_tf(points, dense_tensor_shape, data_format):
    x = points
    assert data_format in ['channels_last', 'channels_first']
    # Add one channel (channels_last convention)
    if data_format == 'channels_last':
        x = tf.pad(x, [[0, 0], [0, 1]])
    else:
        x = tf.pad(x, [[0, 0], [1, 0]])
    st = tf.sparse.SparseTensor(x, tf.ones_like(x[:, 0]), dense_tensor_shape)
    return st


def process_x(x, dense_tensor_shape):
    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)
    return x


def quantize_tensor(x):
    x = tf.clip_by_value(x, 0, 1)
    x = tf.round(x)
    x = tf.cast(x, tf.uint8)
    return x


def input_fn(points, batch_size, dense_tensor_shape, data_format, repeat=True, shuffle=True, prefetch_size=1):
    # Create input data pipeline.
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_generator(lambda: iter(points), tf.int64, tf.TensorShape([None, 3]))
        if shuffle:
            dataset = dataset.shuffle(len(points))
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.map(lambda x: pc_to_tf(x, dense_tensor_shape, data_format))
        dataset = dataset.map(lambda x: process_x(x, dense_tensor_shape))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)

    return dataset


def v1_summaries(train_loss, mbpov_y, mbpov_total, train_fl, log_y_likelihoods, num_occupied_voxels, x, x_tilde,
                 x_tilde_quant, y, y_likelihoods, y_tilde):
    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("mbpov/y", mbpov_y)
    tf.summary.scalar("mbpov/total", mbpov_total)
    tf.summary.scalar("fl", train_fl)
    tf.summary.scalar("num_occupied_voxels", num_occupied_voxels)
    tf.summary.histogram("y", y)
    tf.summary.histogram("y_tilde", y_tilde)
    tf.summary.histogram("x", x)
    tf.summary.histogram("x_tilde", x_tilde)
    tf.summary.histogram("x_tilde_quant", x_tilde_quant)
    tf.summary.histogram("y_likelihoods", y_likelihoods)
    tf.summary.histogram("log_y_likelihoods", log_y_likelihoods)


def v2_summaries(log_z_likelihoods, sigma_tilde, train_mbpov_z, z, z_likelihoods, z_tilde):
    tf.summary.histogram("z", z)
    tf.summary.histogram("z_tilde", z_tilde)
    tf.summary.scalar("mbpov/z", train_mbpov_z)
    tf.summary.histogram("sigma_tilde", sigma_tilde)
    tf.summary.histogram("z_likelihoods", z_likelihoods)
    tf.summary.histogram("log_z_likelihoods", log_z_likelihoods)


def binary_classification_summaries(x_quant, x_tilde_quant):
    tp = tf.count_nonzero(x_tilde_quant * x_quant, dtype=tf.float32)
    tn = tf.count_nonzero((x_tilde_quant - 1) * (x_quant - 1), dtype=tf.float32)
    fp = tf.count_nonzero(x_tilde_quant * (x_quant - 1), dtype=tf.float32)
    fn = tf.count_nonzero((x_tilde_quant - 1) * x_quant, dtype=tf.float32)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    f1_score = (2 * precision * recall) / (precision + recall)

    tf.summary.scalar("bc/precision", precision)
    tf.summary.scalar("bc/recall", recall)
    tf.summary.scalar("bc/accuracy", accuracy)
    tf.summary.scalar("bc/specificity", specificity)
    tf.summary.scalar("bc/f1_score", f1_score)


def sparse_to_dense(block, x_shape, data_format):
    x_val = np.zeros(x_shape, dtype=np.float32)
    if data_format == 'channels_first':
        x_val[0, 0, block[:, 0], block[:, 1], block[:, 2]] = 1.0
    else:
        x_val[0, block[:, 0], block[:, 1], block[:, 2], 0] = 1.0
    return x_val


def get_normals_if(x, with_normals):
    return x[:, x.shape[1]-3:x.shape[1]] if with_normals else None


def add_channels(shape, channels, data_format):
    if data_format == 'channels_first':
        return tf.concat([[channels], shape], 0)
    else:
        return tf.concat([shape, [channels]], 0)


def select_best_per_opt_metric(binstr, x_hat_list, level, opt_metrics, points, resolution, with_normals,
                               opt_groups=('d1', 'd2')):
    """Selects best opt_metric for each opt_group

    :param binstr: octree binstr specification
    :param x_hat_list: a list of list of blocks for each opt_metric
    :param level: octree partitioning level
    :param opt_metrics: list of opt_metrics names
    :param points: original points for comparison
    :param resolution: original point cloud resolution
    :param with_normals: whether normals are available or not
    :param opt_groups: opt_metric prefixes used for grouping
    :return: metadata regarding best selections
    """
    assert len(opt_metrics) == len(x_hat_list), f'lengths of opt_metrics {len(opt_metrics)} and x_hat_list' +\
                                                f' {len(x_hat_list)} should be equal'
    # opt_metric groups
    om_groups = [[(x, y, i) for i, (x, y) in enumerate(zip(opt_metrics, x_hat_list))
                  if x.startswith(group)] for group in opt_groups]
    bbox_min = [0, 0, 0]
    # Assume blocks of equal size
    bbox_max = [resolution] * 3
    t1 = cKDTree(points[:, :3])
    metadata = []
    logger.info(f'Processing metrics {opt_metrics} with groups {opt_groups}')
    for group, om_group in zip(opt_groups, om_groups):
        metric_key = f'{group}_psnr'
        if len(om_group) == 0:
            logger.info(f'Group {group} : {metric_key} no data')
            continue
        om_names, cur_x_hat_list, indexes = zip(*om_group)

        cur_blocks_depart = [departition_octree(x, binstr, bbox_min, bbox_max, level) for x in cur_x_hat_list]
        cur_blocks_full = [np.vstack(x) for x in cur_blocks_depart]
        cur_metrics_full = [compute_metrics(points[:, :3], x, resolution - 1, p1_n=get_normals_if(points, with_normals),
                                            t1=t1) for x in cur_blocks_full]
        cur_metrics = [x[metric_key] for x in cur_metrics_full]
        local_best_idx = np.argmax(cur_metrics)
        best_idx = indexes[local_best_idx]
        data = {'idx': best_idx,
                'metrics': cur_metrics_full[local_best_idx],
                'x_hat_list': cur_x_hat_list[local_best_idx],
                'blocks_depart': cur_blocks_depart[local_best_idx],
                'blocks_full': cur_blocks_full[local_best_idx]}
        results_dict = dict(zip(om_names, [f"{x:.2f}" for x in cur_metrics]))
        logger.info(f'Group {group} : {metric_key} best idx {best_idx} {opt_metrics[best_idx]}\n' +
                    pprint.pformat(results_dict))
        metadata.append(data)
    return metadata


class CompressionModel:
    def __init__(self, n_thresholds=2 ** 8, data_format='channels_first'):
        self.thresholds = np.linspace(0, 1.0, n_thresholds)
        self.data_format = data_format

    def compress_blocks(self, sess, blocks, binstr, points, resolution, level, with_normals=False,
                        opt_metrics=('d1_mse',), max_deltas=(np.inf,), fixed_threshold=False, debug=False):
        """Uses the compression model to compress a point cloud"""
        strings_list = []
        threshold_list = []
        debug_t_list = []
        x_hat_list = []
        no_op = tf.no_op()
        for j, block in enumerate(blocks):
            logger.info(f'Compress block {j}/{len(blocks)}: start')
            block_uint32 = block.astype(np.uint32)
            x_val = sparse_to_dense(block_uint32, self.x.shape, self.data_format)
            logger.info('Compress block: run session')
            fetches = [self.strings, self.x_hat, self.debug_tensors if debug else no_op]
            strings, x_hat, debug_tensors = sess.run(fetches, feed_dict={self.x: x_val})
            logger.info('Compress block: session done')
            strings = [s[0] for s in strings]
            x_hat = x_hat[0, 0, :, :, :] if self.data_format == 'channels_first' else x_hat[0, :, :, :, 0]
            x_hat = np.clip(x_hat, 0.0, 1.0)
            logger.info('Compress block: compute optimal thresholds')
            normals = get_normals_if(block, with_normals)
            opt_metrics_ret, best_thresholds = compute_optimal_thresholds(block, x_hat, self.thresholds, resolution,
                                                                          normals=normals, opt_metrics=opt_metrics,
                                                                          max_deltas=max_deltas, fixed_threshold=fixed_threshold)
            logger.info('Compress block: done')
            x_hat_list.append([np.argwhere(x_hat > self.thresholds[t]).astype(np.float32) for t in best_thresholds])
            strings_list.append(strings)
            threshold_list.append(best_thresholds)
            debug_t_list.append(debug_tensors)
        # block -> opt metric to opt metric -> block
        threshold_list = list(zip(*threshold_list))
        x_hat_list = list(zip(*x_hat_list))
        metadata = select_best_per_opt_metric(binstr, x_hat_list, level, opt_metrics_ret, points, resolution, with_normals)
        data_list = [list(zip(strings_list, threshold_list[x['idx']])) for x in metadata]
        return data_list, metadata, debug_t_list

    def decompress_blocks(self, sess, blocks, x_shape, debug=False):
        """Uses the decompression model to decompress a point cloud"""
        dec_blocks = []
        debug_t_list = []
        for i, (strings, best_threshold_idx) in enumerate(blocks):
            logger.info(f'Decompress block {i}/{len(blocks)}: start')
            strings = [[s] for s in strings]
            threshold = self.thresholds[best_threshold_idx]
            logger.info(f'Decompress block {i}/{len(blocks)}: run session')
            fetches = [self.x_hat, self.debug_tensors if debug else tf.no_op()]
            x_hat, debug_tensors = sess.run(fetches, feed_dict={self.x_shape_t: x_shape, **dict(zip(self.strings_t, strings))})
            logger.info(f'Decompress block {i}/{len(blocks)}: session done')
            x_hat = x_hat[0, 0, :, :, :] if self.data_format == 'channels_first' else x_hat[0, :, :, :, 0]
            x_hat = x_hat > threshold
            pa = np.argwhere(x_hat).astype('float32')
            logger.info(f'Decompress block {i}/{len(blocks)}: done')
            dec_blocks.append(pa)
            debug_t_list.append(debug_tensors)
        return dec_blocks, debug_t_list


class CompressionModelV1(CompressionModel):
    def __init__(self, num_filters=32,
                 analysis_transform_type=TransformType.AnalysisTransformV1,
                 synthesis_transform_type=TransformType.SynthesisTransformV1, *args, **kwargs):
        self.num_filters = num_filters
        self.analysis_transform_class = analysis_transform_type.value
        self.synthesis_transform_class = synthesis_transform_type.value
        super(CompressionModelV1, self).__init__(*args, **kwargs)

    def train(self, x, gamma, alpha, lmbda):
        """Initializes the training model"""
        analysis_transform = self.analysis_transform_class(self.num_filters, data_format=self.data_format)
        synthesis_transform = self.synthesis_transform_class(self.num_filters, data_format=self.data_format)
        entropy_bottleneck = tfc.EntropyBottleneck(data_format=self.data_format)

        # Build autoencoder.
        y = analysis_transform(x)
        y_tilde, y_likelihoods = entropy_bottleneck(y, training=True)
        x_tilde = synthesis_transform(y_tilde)
        x_quant = quantize_tensor(x)
        x_tilde_quant = quantize_tensor(x_tilde)

        # Loss
        num_occupied_voxels = tf.reduce_sum(x)
        log_y_likelihoods = tf.log(y_likelihoods)
        self.train_mbpov = tf.reduce_sum(log_y_likelihoods) / (-np.log(2) * num_occupied_voxels)
        self.train_fl = focal_loss(x, x_tilde, gamma=gamma, alpha=alpha)
        self.train_loss = lmbda * self.train_fl + self.train_mbpov

        v1_summaries(self.train_loss, self.train_mbpov, self.train_mbpov, self.train_fl, log_y_likelihoods,
                     num_occupied_voxels, x, x_tilde, x_tilde_quant, y, y_likelihoods, y_tilde)
        binary_classification_summaries(x_quant, x_tilde_quant)
        self.merged_summary = tf.summary.merge_all()

        # Minimize loss and auxiliary loss, and execute update op.
        self.step = tf.train.get_or_create_global_step()
        main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        main_step = main_optimizer.minimize(self.train_loss, global_step=self.step)
        aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
        self.train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    def compress(self, x_shape):
        """Initializes the compression model"""
        analysis_transform = self.analysis_transform_class(self.num_filters, data_format=self.data_format)
        synthesis_transform = self.synthesis_transform_class(self.num_filters, data_format=self.data_format)
        entropy_bottleneck = tfc.EntropyBottleneck(data_format=self.data_format)

        self.x = tf.placeholder(tf.float32, shape=x_shape)
        y = analysis_transform(self.x)
        y_string = entropy_bottleneck.compress(y)
        y_hat = entropy_bottleneck.decompress(y_string, tf.shape(y)[1:], channels=self.num_filters)
        self.x_hat = synthesis_transform(y_hat)
        self.strings = (y_string,)
        self.debug_tensors = {'y_hat': y_hat, 'x_hat': self.x_hat}

    def decompress(self):
        """Initializes the decompression model"""
        synthesis_transform = self.synthesis_transform_class(self.num_filters, data_format=self.data_format)
        entropy_bottleneck = tfc.EntropyBottleneck(data_format=self.data_format)

        y_string_t = tf.placeholder(tf.string)
        self.strings_t = [y_string_t]
        self.x_shape_t = tf.placeholder(tf.int32, shape=(3,))
        y_shape_t = add_channels(self.x_shape_t // 8, self.num_filters, data_format=self.data_format)
        y_hat = entropy_bottleneck.decompress(y_string_t, y_shape_t, channels=self.num_filters)
        x_hat = synthesis_transform(y_hat)
        self.x_hat = x_hat
        self.debug_tensors = {'y_hat': y_hat, 'x_hat': self.x_hat}


class CompressionModelV2(CompressionModel):
    def __init__(self, num_filters=32,
                 analysis_transform_type=TransformType.AnalysisTransformV1,
                 synthesis_transform_type=TransformType.SynthesisTransformV1,
                 hyper_analysis_transform_type=TransformType.HyperAnalysisTransform,
                 hyper_synthesis_transform_type=TransformType.HyperSynthesisTransform,
                 scales_min=0.11, scales_max=256, scales_levels=64, *args, **kwargs):
        self.num_filters = num_filters
        self.analysis_transform_class = analysis_transform_type.value
        self.synthesis_transform_class = synthesis_transform_type.value
        self.hyper_analysis_transform_class = hyper_analysis_transform_type.value
        self.hyper_synthesis_transform_class = hyper_synthesis_transform_type.value
        self.scale_table = np.exp(np.linspace(np.log(scales_min), np.log(scales_max), scales_levels))
        super(CompressionModelV2, self).__init__(*args, **kwargs)

    def train(self, x, gamma, alpha, lmbda):
        """Initializes the training model"""
        analysis_transform = self.analysis_transform_class(self.num_filters, data_format=self.data_format)
        synthesis_transform = self.synthesis_transform_class(self.num_filters, data_format=self.data_format)
        hyper_analysis_transform = self.hyper_analysis_transform_class(self.num_filters, data_format=self.data_format)
        hyper_synthesis_transform = self.hyper_synthesis_transform_class(self.num_filters, data_format=self.data_format)
        entropy_bottleneck = tfc.EntropyBottleneck(data_format=self.data_format)

        # Build autoencoder.
        y = analysis_transform(x)
        z = hyper_analysis_transform(y)
        z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
        sigma_tilde = hyper_synthesis_transform(z_tilde)
        conditional_bottleneck = tfc.GaussianConditional(sigma_tilde, self.scale_table)
        y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
        x_tilde = synthesis_transform(y_tilde)
        x_quant = quantize_tensor(x)
        x_tilde_quant = quantize_tensor(x_tilde)

        # Loss
        num_occupied_voxels = tf.reduce_sum(x)
        log_y_likelihoods = tf.log(y_likelihoods)
        log_z_likelihoods = tf.log(z_likelihoods)
        denominator = -np.log(2) * num_occupied_voxels
        train_mbpov_y = tf.reduce_sum(log_y_likelihoods) / denominator
        train_mbpov_z = tf.reduce_sum(log_z_likelihoods) / denominator
        self.train_mbpov = train_mbpov_y + train_mbpov_z
        self.train_fl = focal_loss(x, x_tilde, gamma=gamma, alpha=alpha)
        self.train_loss = lmbda * self.train_fl + self.train_mbpov

        v1_summaries(self.train_loss, train_mbpov_y, self.train_mbpov, self.train_fl, log_y_likelihoods,
                     num_occupied_voxels, x, x_tilde, x_tilde_quant, y, y_likelihoods, y_tilde)
        v2_summaries(log_z_likelihoods, sigma_tilde, train_mbpov_z, z, z_likelihoods, z_tilde)
        binary_classification_summaries(x_quant, x_tilde_quant)
        self.merged_summary = tf.summary.merge_all()

        # Minimize loss and auxiliary loss, and execute update op.
        self.step = tf.train.get_or_create_global_step()
        main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        main_step = main_optimizer.minimize(self.train_loss, global_step=self.step)
        aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
        self.train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    def compress(self, x_shape):
        """Initializes the compression model"""
        analysis_transform = self.analysis_transform_class(self.num_filters, data_format=self.data_format)
        synthesis_transform = self.synthesis_transform_class(self.num_filters, data_format=self.data_format)
        hyper_analysis_transform = self.hyper_analysis_transform_class(self.num_filters, data_format=self.data_format)
        hyper_synthesis_transform = self.hyper_synthesis_transform_class(self.num_filters, data_format=self.data_format)
        entropy_bottleneck = tfc.EntropyBottleneck(data_format=self.data_format)

        self.x = tf.placeholder(tf.float32, shape=x_shape)
        y = analysis_transform(self.x)
        z = hyper_analysis_transform(y)
        z_string = entropy_bottleneck.compress(z)
        z_hat = entropy_bottleneck.decompress(z_string, tf.shape(z)[1:], channels=self.num_filters)
        sigma_hat = hyper_synthesis_transform(z_hat)
        conditional_bottleneck = tfc.GaussianConditional(sigma_hat, self.scale_table, dtype=tf.float32)
        y_string = conditional_bottleneck.compress(y)
        y_hat = conditional_bottleneck.decompress(y_string)
        self.x_hat = synthesis_transform(y_hat)
        self.strings = (y_string, z_string)
        self.debug_tensors = {**{'z_hat': z_hat, 'sigma_hat': sigma_hat}, **conditional_bottleneck.dbg_dec,
                              **{'y_hat': y_hat, 'x_hat': self.x_hat}}

    def decompress(self):
        """Initializes the decompression model"""
        synthesis_transform = self.synthesis_transform_class(self.num_filters, data_format=self.data_format)
        hyper_synthesis_transform = self.hyper_synthesis_transform_class(self.num_filters, data_format=self.data_format)
        entropy_bottleneck = tfc.EntropyBottleneck(data_format=self.data_format)

        y_string_t = tf.placeholder(tf.string)
        z_string_t = tf.placeholder(tf.string)
        self.x_shape_t = tf.placeholder(tf.int32, shape=(3,))
        self.strings_t = [y_string_t, z_string_t]
        z_shape_t = add_channels(self.x_shape_t // 16, self.num_filters, self.data_format)
        z_hat = entropy_bottleneck.decompress(z_string_t, z_shape_t, channels=self.num_filters)
        sigma_hat = hyper_synthesis_transform(z_hat)
        conditional_bottleneck = tfc.GaussianConditional(sigma_hat, self.scale_table, dtype=tf.float32)
        y_hat = conditional_bottleneck.decompress(y_string_t)
        x_hat = synthesis_transform(y_hat)
        self.x_hat = x_hat
        self.debug_tensors = {**{'z_hat': z_hat, 'sigma_hat': sigma_hat}, **conditional_bottleneck.dbg_dec,
                              **{'y_hat': y_hat, 'x_hat': self.x_hat}}


class ModelType(Enum):
    v1 = CompressionModelV1
    v2 = CompressionModelV2
