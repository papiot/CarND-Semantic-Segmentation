#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # Reference classroom FCN-8 Encoder and Decoder and the paper mentioned in the Q&A

    weights_init_stddev = 0.01
    weights_regularized_l2 = 1e-3 # taken from the Q&A

    # Convolution 1x1 vgg layer 7
    conv_7 = tf.layers.conv2d(vgg_layer7_out,
                              num_classes,
                              kernel_size=1,
                              strides=(1,1),
                              padding = 'same',
                              kernel_initializer = tf.random_normal_initializer(stddev=weights_init_stddev),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                              name='conv_7')

    # Deconvolution vgg layer 7
    deconv_7 = tf.layers.conv2d_transpose(conv_7,
                                          num_classes,
                                          kernel_size=4,
                                          strides=(2,2),
                                          padding='same',
                                          kernel_initializer = tf.random_normal_initializer(stddev=weights_init_stddev),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                          name='deconv_7')


    # Convolution 1x1 vgg layer 4
    conv_4 = tf.layers.conv2d(vgg_layer4_out,
                              num_classes,
                              kernel_size=1,
                              strides=(1,1),
                              padding='same',
                              kernel_initializer = tf.random_normal_initializer(stddev=weights_init_stddev),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                              name='conv_4')

    # Skip Layer 4
    skip_layer_4 = tf.add(deconv_7, conv_4, name='skip_layer_4')

    # Deconvolution vgg layer 4
    deconv_4 = tf.layers.conv2d_transpose(skip_layer_4,
                                          num_classes,
                                          kernel_size=4,
                                          strides=(2,2),
                                          padding='same',
                                          kernel_initializer = tf.random_normal_initializer(stddev=weights_init_stddev),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                          name='deconv_4')


    # Convolution 1x1 vgg layer 3
    conv_3 = tf.layers.conv2d(vgg_layer3_out,
                              num_classes,
                              kernel_size=1,
                              strides=(1,1),
                              padding='same',
                              kernel_initializer = tf.random_normal_initializer(stddev=weights_init_stddev),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                              name='conv_3')

    # Skip Layer 3
    skip_layer_3 = tf.add(deconv_4, conv_3, name='skip_layer_3')

    # Deconvolution vgg layer 3
    deconv_3 = tf.layers.conv2d_transpose(skip_layer_3,
                                          num_classes,
                                          kernel_size=16, # Changed from 4 to 16
                                          strides=(8,8),
                                          padding='same',
                                          kernel_initializer = tf.random_normal_initializer(stddev=weights_init_stddev),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                          name='deconv_3')


    return deconv_3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function


    # See FCN-8 Classification and Loss lesson
    # 2d tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    label = tf.reshape(correct_label, (-1, num_classes))

    # Loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))

    # Optimizer - Adam optimizer to have variable learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Apply optimizer to loss function
    train_optimizer = optimizer.minimize(cross_entropy_loss)

    return logits, train_optimizer, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    print('training for {} epochs'.format(epochs))
    print()
    
    for epoch in range(epochs):
        print('epoch: {}'.format(epoch+1))
        loss_log = []
        for image, label in get_batches_fn(batch_size):
            # Training
            _ ,loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={
                                   input_image: image,
                                   correct_label: label,
                                   keep_prob: 0.5,
                                   learning_rate: 0.00001 # try 0.0005
                               })
            loss_log.append('{:3f}'.format(loss))
        print(loss_log)
        print()
    print('done training')


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

         # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # Placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        # Getting layers from vgg.
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        
        # Creating new layers.
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        
        # Creating loss and optimizer operations.
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        epochs = 12 # 6 12 24 48
        batch_size = 5
        
        saver = tf.train.Saver()

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
       # helper.save_inference_samples(model_dir, runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, saver)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
