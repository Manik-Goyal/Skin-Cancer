from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import pickle
import sys
import os
import time
import datetime


import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import scipy as sp

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from svm_layer import utils as su
tf.flags.DEFINE_string("input_training", "", "Source file for training")
tf.flags.DEFINE_string("model_path", "./neural_ensemble/model.ckpt", "path to save model")

FLAGS = tf.flags.FLAGS
ids, labels, features = su.read_pickled_data(FLAGS.input_training)
print(np.shape(labels), file=sys.stderr)



# Placeholder variable for the input images# Place 
x = tf.placeholder(tf.float32, shape=[None, 1536*1], name='X')
# Reshape it into [batch, height, width, num_models]
x = tf.reshape(x, [-1, 1536, 1, 2])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 3], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

num_samples = len(ids)

def new_conv_layer(input, num_input_channels, filter_size_h, filter_size_w, num_filters, name, w_name):
    
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size_h, filter_size_w, num_input_channels, num_filters]
        print('input', np.shape(input))

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05), w_name)

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]), 'biases')

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='VALID')
        # Add the biases to the results of the convolution.
        print(w_name, np.shape(weights))
        print('layer', np.shape(layer))
        print('biases', np.shape(biases))
        layer += biases
        
	layer=tf.nn.relu(layer)
        return layer, weights

        
# Convolutional Layer 1x1
layer_conv_1, weights_conv_1 = new_conv_layer(input=x, num_input_channels=2, filter_size_h=1536, filter_size_w=1, num_filters=3, name ="conv1_1x1", w_name='weights_1')

layer_conv_2, weights_conv_2 = new_conv_layer(input=layer_conv_1, num_input_channels=3, filter_size_h=1, filter_size_w=1, num_filters=3, name ="conv2_1x1", w_name='weights_2')

layer_conv=tf.reshape(layer_conv_2,[-1,3])

with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_conv)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
# Use Cross entropy cost function
with tf.name_scope("cross_ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_conv, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    
    
# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
    
# Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
# Initialize the FileWriter
writer = tf.summary.FileWriter("Training_FileWriter/")


# Add the cost and accuracy to summary
tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

num_epochs = 300
batch_size = 500

saver = tf.train.Saver({"weights_1": weights_conv_1, "weights_2": weights_conv_2})
#saver = tf.train.Saver({"weights_1": weights_conv_1})

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    
    # Loop over number of epochs
    for epoch in range(num_epochs):
        
        start_time = time.time()
        train_accuracy = 0
        
        for batch in range(0, int(num_samples/batch_size)):
            
            # Get a batch of images and labels
            x_batch=features[batch_size*batch:batch_size*(batch+1)]
            #print(np.shape(x_batch))
            x_batch = np.reshape(x_batch, [-1, 1536, 1, 2],'F')
            #print(x_batch[0])
            
            y_true_batch=labels[batch_size*batch:batch_size*(batch+1)]
            #print(np.shape(y_true_batch))
            y_true_batch=[int(i) for i in y_true_batch]
            y_true_batch_hot=np.zeros((batch_size,3))
            y_true_batch_hot[np.arange(batch_size),y_true_batch]=1
            #print(y_true_batch_hot)
            
            # Put the batch into a dict with the proper names for placeholder variables
            feed_dict_train = {x: x_batch, y_true: y_true_batch_hot}
            
            # Run the optimizer using this batch of training data.
            sess.run(optimizer, feed_dict=feed_dict_train)
            
            # Calculate the accuracy on the batch of training data
            train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
            
            # Generate summary with the current batch of data and write to file
            summ = sess.run(merged_summary, feed_dict=feed_dict_train)
            writer.add_summary(summ, epoch*int(num_samples/batch_size) + batch)
        
          
        train_accuracy /= int(num_samples/batch_size)


        end_time = time.time()
        
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(train_accuracy))
        
        
    save_path = saver.save(sess, FLAGS.model_path)
    print("Model saved in path: %s" % save_path)
