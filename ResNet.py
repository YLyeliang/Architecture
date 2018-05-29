import tensorflow as tf
import numpy as np
import data_utils
import cv2
import matplotlib.pyplot as plt
import math

# get cifar10 dataset, you should specify the directory fo your cifar10 dataset if you use this funciton.
data=data_utils.get_CIFAR10_data()
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']
print(X_train.shape)


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name = 'x')

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name = 'keep_prob')



def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, name):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0001)
    n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    with tf.variable_scope(name):
        weights = tf.get_variable('conv_weights', shape = [conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
                              initializer = tf.random_normal_initializer(stddev = np.sqrt(2.0 / n)),
                              regularizer = regularizer)
        L = tf.nn.conv2d(x_tensor, weights, strides = [1, conv_strides[0], conv_strides[1], 1], padding = 'SAME')
        L = batch_norm(L,'bn')
        L = relu(L)
    return L

def batch_norm(x_tensor,name=None):
    mean,variance=tf.nn.moments(x_tensor,axes = [0])
    L=tf.nn.batch_normalization(x_tensor,mean,variance,0.01,1,0.001,name = name)
    return L


def maxpool(x_tensor, pool_ksize, pool_strides):
    return tf.nn.max_pool(x_tensor, ksize = [1, pool_ksize[0], pool_ksize[1], 1], strides = [1, pool_strides[0], pool_strides[1], 1], padding = 'VALID')

def avgpool(x_tensor, pool_ksize, pool_strides):
    return tf.nn.avg_pool(x_tensor, ksize = [1, pool_ksize[0], pool_ksize[1], 1], strides = [1, pool_strides[0], pool_strides[1], 1], padding = 'VALID')

def relu(L):
    return tf.nn.relu(L)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    x_shape = x_tensor.get_shape().as_list()
    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0001)

    weights = tf.get_variable('weight', shape = [x_shape[1], num_outputs],
                              initializer = tf.uniform_unit_scaling_initializer(factor = 1.0),
                              regularizer = regularizer)
    bias = tf.Variable(tf.zeros([num_outputs]))
    out = tf.add(tf.matmul(x_tensor, weights), bias)

    return out

# def horizontal_flip(image, axis):
#     '''
#     Flip an image at 50% possibility
#     :param image: a 3 dimensional numpy array representing an image
#     :param axis: 0 for vertical flip and 1 for horizontal flip
#     :return: 3D image after flip
#     '''
#     flip_prop = np.random.randint(low = 0, high = 2)
#     if flip_prop == 0:
#         image = cv2.flip(image, axis)
#
#     return image


# def random_crop_and_flip(batch_data):
#     '''
#     Helper to random crop and random flip a batch of images
#     :param padding_size: int. how many layers of 0 padding was added to each side
#     :param batch_data: a 4D batch array
#     :return: randomly cropped and flipped image
#     '''
#     cropped_batch = np.zeros(len(batch_data) * 32 * 32 * 3).reshape(
#         len(batch_data), 32, 32, 3)
#     padded_batch = np.zeros(len(batch_data) * 40 * 40 * 3).reshape(
#         len(batch_data), 40, 40, 3)
#     padded_batch[:, 4:36, 4:36, :] = batch_data
#
#     for i in range(len(batch_data)):
#         x_offset = np.random.randint(low = 0, high = 2 * 4, size = 1)[0]
#         y_offset = np.random.randint(low = 0, high = 2 * 4, size = 1)[0]
#
#         cropped_batch[i, ...] = padded_batch[i, ...][x_offset:x_offset + 32,
#                                 y_offset:y_offset + 32, :]
#
#         cropped_batch[i, ...] = horizontal_flip(image = cropped_batch[i, ...], axis = 1)
#
#     return cropped_batch
#
#  def generate_augment_train_batch(train_data, train_labels, train_batch_size):
#         '''
#         This function helps generate a batch of train data, and random crop, horizontally flip
#         and whiten them at the same time
#         :param train_data: 4D numpy array
#         :param train_labels: 1D numpy array
#         :param train_batch_size: int
#         :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
#         '''
#         batch_data = random_crop_and_flip(train_data)
#         batch_label = train_labels
#         return batch_data, batch_label


def residual_block(x,filters,name=None):
    with tf.variable_scope(name):
        L=batch_norm(x,'bn')
        L=relu(L)
        L=conv2d(L,filters,(3,3),(1,1),'conv1')
        L=conv2d(L,filters,(3,3),(1,1),'conv2')
        L=L+x
        return L

def residual_blocks(x,n,filters=16,name=None):
    for i in range(n):
        L=residual_block(x,filters = filters,name = name+'{0}'.format(i+1))
        x=L
    return L





def resnet_110(x,keep_prob=None,phase_train=None):
    with tf.variable_scope("resnet_110"):
        conv1 = conv2d(x, 16, (3,3), (1,1), 'conv1')
        # resnet-1-18(32x32)
        with tf.variable_scope('32x32'):
            conv2=residual_blocks(conv1,18,name = 'residual_')

        # resnet-19-36(16x16)
        with tf.variable_scope('16x16'):
            conv3=batch_norm(conv2,name = 'bn_19')
            conv3=relu(conv3)
            conv3=conv2d(conv3,32,(1,1),(2,2),'subsample_1')
            conv3=residual_blocks(conv3,18,filters = 32,name = 'residual_')

        # resnet-36-53(8x8)
        with tf.variable_scope('8x8'):
            conv4=batch_norm(conv3,name = 'bn_36')
            conv4=relu(conv4)
            conv4=conv2d(conv4,64,(1,1),(2,2),'subsample_2')
            conv4=residual_blocks(conv4,18,filters = 64,name = 'residual_')
            conv=batch_norm(conv4,name = 'bn_53')
            conv=relu(conv)

        # global avgpool
        conv=tf.reduce_mean(conv,[1,2])

        out=output(conv,10)
        return out

def run_model(session, predict, loss_val, Xd, yd, epochs = 1, batch_size = 64, print_every = 100, training = None,
              plot_losses = False):
    # Accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [cost, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    lr=None
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []

        # update learning rate
        if e < 79:
            lr = 0.1
        elif e >= 79 and e < 120:
            lr = 0.01
        else:
            lr = 0.001

        # make sure we iterater over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):  # 括号中计算出迭代次数
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {x: Xd[idx, :], y: yd[idx], learning_rate:lr}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict = feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,
                                                                                                             loss,
                                                                                                             np.sum(
                                                                                                                 corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_correct

##############################
## Build the Neural Network ##
##############################
tf.reset_default_graph()
# Remove previous weights, bias, inputs, etc..
x = neural_net_image_input((32, 32, 3))
y = tf.placeholder(tf.int64,[None],name = 'y')
keep_prob = neural_net_keep_prob_input()
training = tf.placeholder(tf.bool, name = 'training')
learning_rate = tf.placeholder(tf.float32)
# Model

logits = resnet_110(x, keep_prob, training)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name = 'logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.one_hot(y,10)))
starter_learning_rate = learning_rate
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, use_nesterov = True, momentum = 0.9).minimize(
    cost)

save_model_path= 'D:/tmp/ResNet/'

saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess, logits, cost, X_train, y_train, 20, 64, 100, optimizer)

# Save Model

save_path = saver.save(sess, save_model_path)

print('Validation')
run_model(sess, logits, cost, X_val, y_val, 1, 64)
