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
import sklearn as sk
import sklearn


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from svm_layer import utils as su

tf.flags.DEFINE_string('input_model',"",'input trained model, in pickle format.')
tf.flags.DEFINE_string("input_data", "", "Source file for training")
tf.flags.DEFINE_string('output_predictions', "" ,'input file with the test data, in isbi challenge format (default=stdout).')
tf.flags.DEFINE_string("output_metrics", "" ,'input file with the test data, in text format (default=stdout).')
tf.flags.DEFINE_string("pool_by_id",'none', "pool answers of contiguous identical ids: none (default), avg, max, xtrm'")

FLAGS = tf.flags.FLAGS
image_ids, labels, features = su.read_pickled_data(FLAGS.input_data)
print(np.shape(labels), file=sys.stderr)



# Placeholder variable for the input images# Place 
x = tf.placeholder(tf.float32, shape=[None, 1536*1], name='X')
# Reshape it into [batch, height, width, num_models]
x = tf.reshape(x, [-1, 1536, 1, 2])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 3], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

num_samples = len(image_ids)

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
        print('bnm', np.shape(weights))
        print('apple', np.shape(layer))
        print('bapple', np.shape(biases))
        layer += biases
        
	layer=tf.nn.relu(layer)
        return layer, weights

        
# Convolutional Layer 1x1
layer_conv_1, weights_conv_1 = new_conv_layer(input=x, num_input_channels=2, filter_size_h=1536, filter_size_w=1, num_filters=3, name ="conv1_1x1", w_name='weights_1')
print('bannaan', layer_conv_1)
layer_conv_2, weights_conv_2 = new_conv_layer(input=layer_conv_1, num_input_channels=3, filter_size_h=1, filter_size_w=1, num_filters=3, name ="conv2_1x1", w_name='weights_2')
layer_conv=tf.reshape(layer_conv_2,[-1,3])

with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_conv)
    y_pred_cls = tf.argmax(y_pred, dimension=1)

saver = tf.train.Saver({"weights_1": weights_conv_1, "weights_2": weights_conv_2})
#saver = tf.train.Saver({"weights_1": weights_conv_1})

pred=[]
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, FLAGS.input_model)

    # Get a batch of images and labels
    x_test=features
    #print(np.shape(x_batch))
    x_test = np.reshape(x_test, [-1, 1536, 1, 2],'F')
    #print(x_batch[0])
            
    y_true_test=labels
    #print(np.shape(y_true_batch))
    y_true_test=[int(i) for i in y_true_test]
    y_true_test_hot=np.zeros((num_samples,3))
    y_true_test_hot[np.arange(num_samples),y_true_test]=1
    #print(y_true_batch_hot)
            
    # Put the batch into a dict with the proper names for placeholder variables
    feed_dict_train = {x: x_test, y_true: y_true_test_hot}
            
    # Run the optimizer using this batch of training data.
    pred=sess.run([y_pred], feed_dict=feed_dict_train)


#print(pred)
pred=np.array(pred[0])
#print(np.shape(pred))

labels_ind=[]
labels_ind.append(labels[0])
image_ids_prev=image_ids[0]
for idx in range(num_samples):
    if image_ids_prev!=image_ids[idx]:
            labels_ind.append(labels[idx])
            
    image_ids_prev=image_ids[idx] 
labels_ind=np.array(labels_ind)

s_all=[]

# "Probabilities" should come between quotes here
# Only if the scores are true logits the probabilities will be consistent
def probability_from_logits(logits) :
    odds = np.exp(logits)
    return odds/(odds+1.0)
def logits_from_probability(prob) :
    with np.errstate(divide='ignore') :
      odds = prob/(1.0-prob)
      return np.log(odds)
def extreme_probability(prob) :
  return prob[np.argmax(np.abs(logits_from_probability(prob)))]

predictions_mel = pred[:,0]
#print(predictions_mel)
#print(np.shape(predictions_mel))
predictions_nv = pred[:,1]
predictions_bkl = pred[:,2]


outfile = open(FLAGS.output_predictions, 'w') if FLAGS.output_predictions else sys.stdout
if FLAGS.pool_by_id=='none' :
  for i in xrange(len(image_ids)) :
    print(image_ids[i], predictions_mel[i], predictions_nv[i], predictions_bkl[i], sep=',', file=outfile)
else :
  previous_id = None
  def print_result() :
    if FLAGS.pool_by_id=='avg' :
      s_all.extend(np.column_stack( (np.mean(all_mel), np.mean(all_nv), np.mean(all_bkl)) ) )
      print(previous_id, np.mean(all_mel), np.mean(all_nv), np.mean(all_bkl), sep=',', file=outfile)
    elif FLAGS.pool_by_id=='max' :
      s_all.extend(np.column_stack( (np.amax(all_mel), np.amax(all_nv), np.amax(all_bkl)) ) )
      print(previous_id, np.amax(all_mel), np.amax(all_nv), np.amax(all_bkl), sep=',', file=outfile)
    elif FLAGS.pool_by_id=='xtrm' :
      s_all.extend(np.column_stack( (extreme_probability(all_mel), extreme_probability(all_nv), extreme_probability(all_bkl)) ) )
      print(previous_id, extreme_probability(all_mel), extreme_probability(all_nv), extreme_probability(all_bkl), sep=',', file=outfile)
    else :
      raise ValueError('Invalid value for FLAGS.pool_by_id: %s' % FLAGS.pool_by_id)

  for i in xrange(len(image_ids)) :
    if image_ids[i]!=previous_id :
      if previous_id is not None :
        print_result()
      previous_id = image_ids[i]
      all_mel = np.asarray([ predictions_mel[i] ])
      all_nv = np.asarray([ predictions_nv[i] ])
      all_bkl = np.asarray([ predictions_bkl[i] ])
    else :
      all_mel = np.concatenate((all_mel, np.asarray([ predictions_mel[i] ])))
      all_nv = np.concatenate((all_nv, np.asarray([ predictions_nv[i] ])))
      all_bkl = np.concatenate((all_bkl, np.asarray([ predictions_bkl[i] ])))
  if previous_id is not None :
    print_result()


metfile = open(FLAGS.output_metrics, 'w') if FLAGS.output_metrics else sys.stderr
try :
  accs = []
  aucs = []
  mAPs = []
  accss=[]
  baccs=[]
  #scores=np.column_stack((predictions_mel, predictions_nv, predictions_bcc, predictions_akiec, predictions_bkl, predictions_df, predictions_vasc))
  #pred=np.array([np.argmax(s) for s in scores])
  #print(s_all,file=metfile)
  p_all=np.array([np.argmax(s) for s in s_all])
  #print(p_all, file=metfile)
  for j, scores_j in [ [0, predictions_mel], [1, predictions_nv], [2, predictions_bkl] ] :
    #labels_j = (labels == j).astype(np.int)
    #acc = sk.metrics.accuracy_score(labels_j, scores_j.astype(np.int))
    #print('Acc[%d]: ' %j, acc, file=metfile)
    #accs.append(acc)
    labels_j = (labels_ind == j).astype(np.int)
    s_all=np.array(s_all)
    scores_j = s_all[:,j]
    #print(scores_j, file=metfile)
    #print(np.shape(scores_j), file=metfile)
    #print(np.shape(labels_j), file=metfile)
    auc = sk.metrics.roc_auc_score(labels_j, scores_j)
    
    print('AUC[%d]: ' % j, auc, file=metfile)

    mAP = sk.metrics.average_precision_score(labels_j, scores_j)
    
    print('mAP[%d]: ' % j, mAP, file=metfile)
    
    #pred_j = (pred == j).astype(np.int)

    #acc_j = np.mean(np.equal(pred_j, labels_j))
    #accss.append(acc_j)
        
    #b_acc_j = np.sum( np.equal(pred_j, 1)& np.equal(labels_j,1) )/np.sum(labels_j)
    #baccs.append(b_acc_j)

    #print('Acc[%d]: ' %j, acc_j, file=metfile)
    #print('Bal_Acc[%d]: '%j, b_acc_j, file=metfile)
    #print('shape ',np.shape(p_all), np.shape(labels_ind), file=metfile)

    pred_j = (p_all == j).astype(np.int) 
    labels_j = (labels_ind == j).astype(np.int)
    
    acc_j = np.mean(np.equal(pred_j, labels_j))
    
        
    b_acc_j = np.sum( np.equal(pred_j, 1)& np.equal(labels_j,1) )/np.sum(labels_j)

    if j!=1:
	mAPs.append(mAP)
	aucs.append(auc)
    	baccs.append(b_acc_j)
        accss.append(acc_j)
    print('Acc[%d]: ' %j, acc_j, file=metfile)
    print('Bal_Acc[%d]: '%j, b_acc_j, file=metfile)
    print('Confusion Matrix:\n', sklearn.metrics.confusion_matrix(labels_j, pred_j), file=metfile)
  
  print('AUC_avg: ', sum(aucs) / 2.0, file=metfile)
  print('mAP_avg: ', sum(mAPs) / 2.0, file=metfile)
  print('Cat_acc: ', sum(accss) / 2.0, file=metfile)
  print('Bal_acc: ', sum(baccs) / 2.0, file=metfile)
  print('Confusion Matrix:\n', sklearn.metrics.confusion_matrix(labels_ind, p_all), file=metfile)
except ValueError :
  pass

