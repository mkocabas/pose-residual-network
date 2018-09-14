import tensorflow as tf
slim = tf.contrib.slim
from ResNet import get_slim_resnet_v1_byname
import collections


def resnet50v1FPN(inputs, is_training, nb_landmarks, weight_decay=0., net_name='resnet_v1_50'):
    """ constructs landmark detection network with resnet_v1_50 and feature pyramid networks
    """
    ## backbone
    logit, end_points = get_slim_resnet_v1_byname('resnet_v1_50', inputs, num_classes=None, is_training=is_training, global_pool=False)
    if net_name == 'resnet_v1_50':
        C_dict = {
            'C2': end_points['resnet_v1_50/block1/unit_2/bottleneck_v1'],  # 120x120
            'C3': end_points['resnet_v1_50/block2/unit_3/bottleneck_v1'],  # 60x60
            'C4': end_points['resnet_v1_50/block3/unit_5/bottleneck_v1'],  # 30x30
            'C5': end_points['resnet_v1_50/block4']  # 15x15
        }
    elif net_name == 'resnet_v1_101':
        C_dict = {
            'C2': end_points['resnet_v1_101/block1/unit_2/bottleneck_v1'],  # 120x120
            'C3': end_points['resnet_v1_101/block2/unit_3/bottleneck_v1'],  # 60x60
            'C4': end_points['resnet_v1_101/block3/unit_22/bottleneck_v1'],  # 30x30
            'C5': end_points['resnet_v1_101/block4']  # 15x15
        }
    else:
        raise Exception('[ERROR] get no feature maps')

    ## build_feature_pyramid: P2(120x120), P3(60x60), P4(30x30), P5(15x15)
    with tf.variable_scope('feature_pyramid'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            stride=1,
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with tf.variable_scope('PFeats'):
                P_dict = collections.OrderedDict()
                # conv1x1(c5) to reduce dim to 256, no need smooth because no add performed.
                P_dict['P5'] = slim.conv2d(C_dict['C5'], 256, kernel_size=[1, 1], scope='P5')

                for layer in range(4, 1, -1):  # 4, 3, 2
                    p, c = P_dict['P' + str(layer + 1)], C_dict['C' + str(layer)]  # p5, c4
                    # upsample p5 to spatial shape of c4
                    up_sample_shape = tf.shape(c)
                    up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]], name='P%d/up_sample' % layer)
                    # conv1x1(c4) to reduce dim to 256
                    c = slim.conv2d(c, 256, kernel_size=[1, 1], scope='P%d/reduce_dimension' % layer)
                    # add(p5, c4) to get p4
                    p = up_sample + c
                    # conv3x3(p4)
                    p = slim.conv2d(p, 256, kernel_size=[3, 3], scope='P%d/avoid_aliasing' % layer)
                    P_dict['P' + str(layer)] = p  # p4

            with tf.variable_scope('DFeats'):
                with slim.arg_scope([slim.conv2d], activation_fn=None):
                    D_dict = collections.OrderedDict()
                    target_HW = P_dict['P2'].shape[1]  # 120x120
                    for i in range(2, 6): # 2,3,4,5
                        # dim 256 -> dim 128
                        d = slim.conv2d(P_dict['P'+str(i)], 128, kernel_size=[3, 3], scope='D'+str(i)+'/conv1')
                        d = slim.conv2d(d, 128, kernel_size=[3, 3], scope='D'+str(i)+'/conv2')

                        # spatial resolution -> 120
                        up_sample = tf.image.resize_nearest_neighbor(d, [target_HW, target_HW], name='D%d/up_sample'%i)
                        D_dict['D'+str(i)] = up_sample
                    # concat
                    D = tf.concat(D_dict.values(), axis=-1, name='concat')  # Nonex120x120x512
                    # smooth 
                    import pdb;pdb.set_trace()
                    D = slim.conv2d(D, 512, kernel_size=[3, 3], activation_fn=tf.nn.relu, scope='smooth')
                    # f 
                    D = slim.conv2d(D, nb_landmarks, kernel_size=[1, 1], scope='final_DFeat')

    return P_dict, D

def test():
    import pdb;pdb.set_trace()
    inputs = tf.placeholder(tf.float32, (None, 480, 480, 1))
    is_training = True
    P_dict, D = resnet50v1FPN(inputs, is_training, 2)

if __name__ == '__main__':
    test()
