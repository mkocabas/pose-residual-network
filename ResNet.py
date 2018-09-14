from tensorflow.contrib.slim.python.slim.nets import resnet_v1, resnet_v2, resnet_utils
import tensorflow as tf
slim = tf.contrib.slim


def resNet(images, is_training=True, reuse=False, scope=None):
    """Constructs network based on resnet_v1_50.
    Args:
      images: A tensor of size [batch, height, width, channels].
      weight_decay: The parameters for weight_decay regularizer.
      is_training: Whether or not in training mode.
      reuse: Whether or not the layer and its variables should be reused.
    Returns:
      feature_map: Features extracted from the model, which are not l2-normalized.
    """
    # Construct Resnet50 features.
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0001)):
        block = resnet_v1.resnet_v1_block
        blocks = [ block('block1', base_depth=64,  num_units=3, stride=2),
                   block('block2', base_depth=128, num_units=4, stride=2),
                   block('block3', base_depth=256, num_units=6, stride=1),
                   block('block4', base_depth=512, num_units=3, stride=1)]

        x30, end_points = resnet_v1.resnet_v1(images, blocks, is_training=is_training,
                global_pool=False, reuse=reuse, scope=scope, include_root_block=True)
    
    x60 = end_points[scope+'/block1'] 
    x60 = slim.conv2d(x60,  64,  [1, 1], 1, padding='SAME', activation_fn=None, reuse=reuse, scope='conv2d_final_x60')

    x30 = slim.conv2d(x30, 512,  [1, 1], 1, padding='SAME', activation_fn=None, reuse=reuse, scope='conv2d_final_x30')

    # get layer outputs we want
    end_points_ = {}
    #  end_points_ = end_points['resnet_v1_50/block2']
    #  end_points_ = end_points['resnet_v1_50/block3']
    #  end_points_ = end_points['resnet_v1_50/block4'] 
    #  end_points_['x30'] = end_points['resnet_v1_50/final'] 
    end_points_['x60'] = x60
    end_points_['x30'] = x30

    return  end_points_


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=False,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
    """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
    resnet_v2_block = resnet_v2.resnet_v2_block
    blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
    ]

    # custom root block, because original root block will use maxpooling with valid padding.
    with tf.variable_scope(scope) as sc:
        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(inputs, 64, 7, stride=2, scope='conv1') # stride 2
        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='SAME') # stride 2
    return resnet_v2.resnet_v2(net, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=False, reuse=reuse, scope=scope)

def get_slim_resnet_v1_byname(net_name,
                       inputs,
                       num_classes=None,
                       is_training=True,
                       global_pool=True,
                       output_stride=None,
                       weight_decay=0.):
    if net_name == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                        num_classes=num_classes,
                                                        is_training=is_training,
                                                        global_pool=global_pool,
                                                        output_stride=output_stride,
                                                       )

        return logits, end_points
    if net_name == 'resnet_v1_101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                         num_classes=num_classes,
                                                         is_training=is_training,
                                                         global_pool=global_pool,
                                                         output_stride=output_stride,
                                                         )
        return logits, end_points

