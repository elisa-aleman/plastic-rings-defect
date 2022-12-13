import os.path
import sys
PythonPath = os.path.join(os.path.expanduser('~'),'PlasticRings', 'python')
sys.path.append(os.path.abspath(PythonPath))
from PlasticMethods import *
import numpy
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf

'''
Alexnet:

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)

Tflearn code referenced from
https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
'''



#### Model defining
def Convolutional_Model_Alexnet_gray(img_height,img_width,color_channels, n_classes=2, run_id='', gray_graph=tf.Graph()):
    # data size
    pixels = img_height*img_width
    batch_size = 64
    # pool window sizes
    pool_1_window_size = 3
    pool_2_window_size = 3
    pool_3_window_size = 3
    # conv window sizes
    conv_1_window_size = 11
    conv_2_window_size = 5
    conv_3_1_window_size = 3
    conv_3_2_window_size = 3
    conv_3_3_window_size = 3
    
    # pool stride sizes
    pool_1_strides = 2
    pool_2_strides = 2
    pool_3_strides = 2
    # conv stride sizes
    conv_1_strides = 4
    conv_2_strides = None #Default
    conv_3_1_strides = None #Default
    conv_3_2_strides = None #Default
    conv_3_3_strides = None #Default
    # compressed data size
    compressed_img_height = img_height/pool_1_window_size/pool_2_window_size
    compressed_img_width = img_width/pool_1_window_size/pool_2_window_size
    # nodes
    n_nodes_conv_layer_1 = 96
    n_nodes_conv_layer_2 = 256
    n_nodes_conv_layer_3_1 = 384
    n_nodes_conv_layer_3_2 = 384
    n_nodes_conv_layer_3_3 = 256
    n_nodes_fc_layer_4 = 4096
    n_nodes_fc_layer_5 = 4096
    # input changes for fully connected
    n_inputs_fc_layer_3 = compressed_img_width*compressed_img_height*n_nodes_conv_layer_2
    #
    with gray_graph.as_default():
        # Input Layer
        gray_convnet = input_data(shape=[None,img_width,img_height,color_channels], name='{}_input'.format(run_id))
        # Convolution - Pool Layer 1
        gray_convnet = conv_2d(gray_convnet, n_nodes_conv_layer_1, conv_1_window_size, strides=conv_1_strides, activation='relu')
        gray_convnet = max_pool_2d(gray_convnet, pool_1_window_size, strides=pool_1_strides)
        gray_convnet = local_response_normalization(gray_convnet)
        # Convolution - Pool Layer 2
        gray_convnet = conv_2d(gray_convnet, n_nodes_conv_layer_2, conv_2_window_size, activation='relu')
        gray_convnet = max_pool_2d(gray_convnet, pool_2_window_size, strides=pool_2_strides)
        gray_convnet = local_response_normalization(gray_convnet)
        # 3 Convolutions 1 Pool Layer 3
        gray_convnet = conv_2d(gray_convnet, n_nodes_conv_layer_3_1, conv_3_1_window_size, activation='relu')
        gray_convnet = conv_2d(gray_convnet, n_nodes_conv_layer_3_2, conv_3_2_window_size, activation='relu')
        gray_convnet = conv_2d(gray_convnet, n_nodes_conv_layer_3_3, conv_3_3_window_size, activation='relu')
        gray_convnet = max_pool_2d(gray_convnet, pool_3_window_size, strides=pool_3_strides)
        gray_convnet = local_response_normalization(gray_convnet)
        # Fully connected layer 4
        gray_convnet = fully_connected(gray_convnet, n_nodes_fc_layer_4, activation='tanh')
        gray_convnet = dropout(gray_convnet, 0.5) # 50% keep rate
        # Fully connected layer 4
        gray_convnet = fully_connected(gray_convnet, n_nodes_fc_layer_5, activation='tanh')
        gray_convnet = dropout(gray_convnet, 0.5) # 50% keep rate
        ###
        # Output layer
        gray_convnet = fully_connected(gray_convnet, n_classes, activation='softmax')
        gray_convnet = regression(
            gray_convnet, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        # Set the model to follow this network
        # tensorboard_verbose = 0: Loss, Accuracy (Best Speed)
        tensorboard_log = MakeLogFile('')
        gray_model = tflearn.DNN(gray_convnet, tensorboard_dir=tensorboard_log)
    return gray_model, gray_graph
#### Model defining
def Convolutional_Model_Alexnet_edge(img_height,img_width,color_channels, n_classes=2, run_id='', edge_graph=tf.Graph()):
    # data size
    pixels = img_height*img_width
    batch_size = 64
    # pool window sizes
    pool_1_window_size = 3
    pool_2_window_size = 3
    pool_3_window_size = 3
    # conv window sizes
    conv_1_window_size = 11
    conv_2_window_size = 5
    conv_3_1_window_size = 3
    conv_3_2_window_size = 3
    conv_3_3_window_size = 3
    
    # pool stride sizes
    pool_1_strides = 2
    pool_2_strides = 2
    pool_3_strides = 2
    # conv stride sizes
    conv_1_strides = 4
    conv_2_strides = None #Default
    conv_3_1_strides = None #Default
    conv_3_2_strides = None #Default
    conv_3_3_strides = None #Default
    # compressed data size
    compressed_img_height = img_height/pool_1_window_size/pool_2_window_size
    compressed_img_width = img_width/pool_1_window_size/pool_2_window_size
    # nodes
    n_nodes_conv_layer_1 = 96
    n_nodes_conv_layer_2 = 256
    n_nodes_conv_layer_3_1 = 384
    n_nodes_conv_layer_3_2 = 384
    n_nodes_conv_layer_3_3 = 256
    n_nodes_fc_layer_4 = 4096
    n_nodes_fc_layer_5 = 4096
    # input changes for fully connected
    n_inputs_fc_layer_3 = compressed_img_width*compressed_img_height*n_nodes_conv_layer_2
    #
    with edge_graph.as_default():
        # Input Layer
        edge_convnet = input_data(shape=[None,img_width,img_height,color_channels], name='{}_input'.format(run_id))
        # Convolution - Pool Layer 1
        edge_convnet = conv_2d(edge_convnet, n_nodes_conv_layer_1, conv_1_window_size, strides=conv_1_strides, activation='relu')
        edge_convnet = max_pool_2d(edge_convnet, pool_1_window_size, strides=pool_1_strides)
        edge_convnet = local_response_normalization(edge_convnet)
        # Convolution - Pool Layer 2
        edge_convnet = conv_2d(edge_convnet, n_nodes_conv_layer_2, conv_2_window_size, activation='relu')
        edge_convnet = max_pool_2d(edge_convnet, pool_2_window_size, strides=pool_2_strides)
        edge_convnet = local_response_normalization(edge_convnet)
        # 3 Convolutions 1 Pool Layer 3
        edge_convnet = conv_2d(edge_convnet, n_nodes_conv_layer_3_1, conv_3_1_window_size, activation='relu')
        edge_convnet = conv_2d(edge_convnet, n_nodes_conv_layer_3_2, conv_3_2_window_size, activation='relu')
        edge_convnet = conv_2d(edge_convnet, n_nodes_conv_layer_3_3, conv_3_3_window_size, activation='relu')
        edge_convnet = max_pool_2d(edge_convnet, pool_3_window_size, strides=pool_3_strides)
        edge_convnet = local_response_normalization(edge_convnet)
        # Fully connected layer 4
        edge_convnet = fully_connected(edge_convnet, n_nodes_fc_layer_4, activation='tanh')
        edge_convnet = dropout(edge_convnet, 0.5) # 50% keep rate
        # Fully connected layer 4
        edge_convnet = fully_connected(edge_convnet, n_nodes_fc_layer_5, activation='tanh')
        edge_convnet = dropout(edge_convnet, 0.5) # 50% keep rate
        ###
        # Output layer
        edge_convnet = fully_connected(edge_convnet, n_classes, activation='softmax')
        edge_convnet = regression(
            edge_convnet, 
            optimizer='momentum', 
            learning_rate=0.001,
            loss='categorical_crossentropy',
            name='{}_targets'.format(run_id)
            )
        # Set the model to follow this network
        # tensorboard_verbose = 0: Loss, Accuracy (Best Speed)
        tensorboard_log = MakeLogFile('')
        edge_model = tflearn.DNN(edge_convnet, tensorboard_dir=tensorboard_log)
    return edge_model, edge_graph

def Train_gray(mode='grayscale_crop_600_resize_240', gray_graph=tf.Graph(), test_size=200):
    ## Data reorganize
    X,Y,test_x,test_y = LoadData(mode=mode, test_size=test_size)
    img_height = len(X[0])
    img_width = len(X[0][0])
    if type(X[0][0][0])==type(numpy.array([])):
        color_channels = len(X[0][0][0])
    else:
        color_channels = 1
    X = numpy.transpose(X, [0,2,1])
    test_x = numpy.transpose(test_x, [0,2,1])
    X = X.reshape([-1,img_width,img_height,color_channels])
    test_x = test_x.reshape([-1,img_width,img_height,color_channels])
    n_classes = len(Y[0])
    ##
    version = 1
    times_run = 1
    times_10x = 1
    batch_size = None
    model_name = 'PlasticRings_{}__v{}'.format(mode,version)
    run_id = '{}_run{}'.format(model_name,times_run)
    while os.path.exists(os.path.join(MakeLogFile(''), run_id)):
        times_run += 1
        run_id = '{}_run{}'.format(model_name,times_run)
    model_path = os.path.abspath(MakeModelPath('')+'/{0}/{0}.tfl'.format(run_id))
    model_dir = os.path.abspath(MakeModelPath('')+'/{0}'.format(run_id))
    gray_model, gray_graph = Convolutional_Model_Alexnet_gray(img_height,img_width,color_channels,n_classes=n_classes,run_id=run_id, gray_graph=gray_graph)
    with gray_graph.as_default():
        gray_model.fit(
            {'{}_input'.format(run_id): X},
            {'{}_targets'.format(run_id): Y},
            n_epoch=10,
            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
            show_metric=True,
            batch_size=batch_size,
            shuffle=False,
            snapshot_step=None,
            run_id=run_id
            )
        gray_model.save(model_path)
    print_log_instructions()

def Train_edge(mode='edges_crop_600_resize_240', edge_graph=tf.Graph(), test_size=200):
    ## Data reorganize
    X,Y,test_x,test_y = LoadData(mode=mode, test_size=test_size)
    img_height = len(X[0])
    img_width = len(X[0][0])
    if type(X[0][0][0])==type(numpy.array([])):
        color_channels = len(X[0][0][0])
    else:
        color_channels = 1
    X = numpy.transpose(X, [0,2,1])
    test_x = numpy.transpose(test_x, [0,2,1])
    X = X.reshape([-1,img_width,img_height,color_channels])
    test_x = test_x.reshape([-1,img_width,img_height,color_channels])
    n_classes = len(Y[0])
    ##
    version = 1
    times_run = 1
    times_10x = 1
    batch_size = None
    model_name = 'PlasticRings_{}__v{}'.format(mode,version)
    run_id = '{}_run{}'.format(model_name,times_run)
    while os.path.exists(os.path.join(MakeLogFile(''), run_id)):
        times_run += 1
        run_id = '{}_run{}'.format(model_name,times_run)
    model_path = os.path.abspath(MakeModelPath('')+'/{0}/{0}.tfl'.format(run_id))
    model_dir = os.path.abspath(MakeModelPath('')+'/{0}'.format(run_id))
    edge_model, edge_graph = Convolutional_Model_Alexnet_edge(img_height,img_width,color_channels,n_classes=n_classes,run_id=run_id, edge_graph=edge_graph)
    with edge_graph.as_default():
        edge_model.fit(
            {'{}_input'.format(run_id): X},
            {'{}_targets'.format(run_id): Y},
            n_epoch=10,
            validation_set=({'{}_input'.format(run_id):test_x},{'{}_targets'.format(run_id):test_y}),
            show_metric=True,
            batch_size=batch_size,
            shuffle=False,
            snapshot_step=None,
            run_id=run_id
            )
        edge_model.save(model_path)
    print_log_instructions()

def main():
    # Train_gray()
    Train_edge()
    
if __name__ == '__main__':
    main()





        