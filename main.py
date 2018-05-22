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


def load_vgg(session, vgg_path):
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
    # VGG saved model is loaded using the tf.saved_model.loader command
    tf.saved_model.loader.load(session, [vgg_tag], vgg_path)
    gph_path = tf.get_default_graph()
    input_image = gph_path.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_value = gph_path.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = gph_path.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = gph_path.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = gph_path.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_image, keep_prob_value, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
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
    #Tf variable
    k_reg=tf.contrib.layers.l2_regularizer(1e-3)
    k_init=tf.random_normal_initializer(stddev=0.01)
    
    # Getting 1 x 1 convolution for VGG layer 3,4,7 so that we have same depth while combining the output from the different layer
    layer7_1X1= tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= k_init,
                                   kernel_regularizer=k_reg )
    layer4_1X1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                   padding='same',
                                   kernel_initializer= k_init,   
                                   kernel_regularizer=k_reg)
    layer3_1X1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                   padding='same',
                                   kernel_initializer= k_init,   
                                   kernel_regularizer=k_reg)
    #Using the transpose - to upsample the 1x1 conv for layer 7
    transposed_layer7 = tf.layers.conv2d_transpose(layer7_1X1, num_classes, 4, 
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= k_init, 
                                             kernel_regularizer= k_reg)
     # skip connection - doing the element-wise addition
    skip_output = tf.add(transposed_layer7, layer4_1X1)
    # upsampling the skip output
    transpose_skip_output = tf.layers.conv2d_transpose(skip_output, num_classes, 4,  
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= k_init, 
                                             kernel_regularizer= k_reg)
    #Skip connection - adding the conv 1x1 of 3rd layer to the previous output
    final_layer_addition = tf.add(transpose_skip_output, layer3_1X1)
    #final upsampling
    final_transpose = tf.layers.conv2d_transpose(final_layer_addition, num_classes, 16,  
                                               strides= (8, 8), 
                                               padding= 'same', 
                                               kernel_initializer= k_init, 
                                               kernel_regularizer= k_reg)
    return final_transpose
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
    #Getting the proper logits for the 2D tensor wherein the row - represents the pixels and the column - represents the class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    update_label = tf.reshape(correct_label, (-1,num_classes))
    # Getting the cross entropy loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= update_label))
    # Getting the optimizer and the  training operation
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    min_optimize = optimizer.minimize(cross_entropy_loss)

    return logits, min_optimize, cross_entropy_loss
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
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())    
    print("Training in process...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
            print("Obtained Loss: = {:.3f}".format(loss))
        print()
    pass
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        batch_size = 2
        epochs = 10
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        img_input, keep_prob, vgg_layer_3, vgg_layer_4, vgg_layer_7 = load_vgg(sess ,vgg_path)
        final_layer = layers(vgg_layer_3, vgg_layer_4, vgg_layer_7, num_classes)
        label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        learning_rate =tf.placeholder(tf.float32, name='learning_rate')
        logits, t_o, c_e_l = optimize(final_layer, label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, t_o, c_e_l, img_input,
             label, keep_prob, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, img_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
