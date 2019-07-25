from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import pickle
import sys

import numpy as np
import scipy as sp
import scipy.stats
import sklearn as sk
import sklearn.decomposition
import sklearn.gaussian_process
import sklearn.model_selection
import sklearn.preprocessing

from svm_layer import utils as su

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Predict the SVM decision.')
parser.add_argument('--input_model', type=str, required=True, help='input trained model, in pickle format.')
parser.add_argument('--input_test', type=str, required=True, help='input file with the test data, in pickle format.')
parser.add_argument('--output_predictions', type=str , help='input file with the test data, in isbi challenge format (default=stdout).')
parser.add_argument('--output_metrics', type=str, help='input file with the test data, in text format (default=stdout).')
parser.add_argument('--pool_by_id', type=str, default='none', help='pool answers of contiguous identical ids: none (default), avg, max, xtrm')
FLAGS = parser.parse_args()

first = start = su.print_and_time('Reading trained model...', file=sys.stderr)
model_file = open(FLAGS.input_model, 'rb')
preprocessor = pickle.load(model_file)
classifier_mel = pickle.load(model_file)
classifier_nv = pickle.load(model_file)
classifier_bcc = pickle.load(model_file)
classifier_akiec = pickle.load(model_file)
classifier_bkl = pickle.load(model_file)
classifier_df = pickle.load(model_file)
classifier_vasc = pickle.load(model_file)
model_file.close()

start = su.print_and_time('Reading test data...',  past=start, file=sys.stderr)
image_ids, labels, features = su.read_pickled_data(FLAGS.input_test)
num_samples = len(image_ids)

labels_ind=[]
labels_ind.append(labels[0])
image_ids_prev=image_ids[0]
for idx in range(num_samples):
    if image_ids_prev!=image_ids[idx]:
            labels_ind.append(labels[idx])
            
    image_ids_prev=image_ids[idx] 
labels_ind=np.array(labels_ind)

start = su.print_and_time('Preprocessing test data...', file=sys.stderr)
features = preprocessor.transform(features)
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

start = su.print_and_time('Predicting test data...\n', past=start, file=sys.stderr)
predictions_mel = probability_from_logits(classifier_mel.decision_function(features))
predictions_nv = probability_from_logits(classifier_nv.decision_function(features))
predictions_bcc = probability_from_logits(classifier_bcc.decision_function(features))
predictions_akiec = probability_from_logits(classifier_akiec.decision_function(features))
predictions_bkl = probability_from_logits(classifier_bkl.decision_function(features))
predictions_df = probability_from_logits(classifier_df.decision_function(features))
predictions_vasc = probability_from_logits(classifier_vasc.decision_function(features))


outfile = open(FLAGS.output_predictions, 'w') if FLAGS.output_predictions else sys.stdout
if FLAGS.pool_by_id=='none' :
  for i in xrange(len(image_ids)) :
    print(image_ids[i], predictions_mel[i], predictions_nv[i], predictions_bcc[i], predictions_akiec[i], predictions_bkl[i], predictions_df[i], predictions_vasc[i], sep=',', file=outfile)
else :
  previous_id = None
  def print_result() :
    if FLAGS.pool_by_id=='avg' :
      s_all.extend(np.column_stack( (np.mean(all_mel), np.mean(all_nv), np.mean(all_bcc), np.mean(all_akiec), np.mean(all_bkl), np.mean(all_df), np.mean(all_vasc)) ) )
      print(previous_id, np.mean(all_mel), np.mean(all_nv), np.mean(all_bcc), np.mean(all_akiec), np.mean(all_bkl), np.mean(all_df), np.mean(all_vasc), sep=',', file=outfile)
    elif FLAGS.pool_by_id=='max' :
      s_all.extend(np.column_stack( (np.amax(all_mel), np.amax(all_nv), np.amax(all_bcc), np.amax(all_akiec), np.amax(all_bkl), np.amax(all_df), np.amax(all_vasc)) ) )
      print(previous_id, np.amax(all_mel), np.amax(all_nv), np.amax(all_bcc), np.amax(all_akiec), np.amax(all_bkl), np.amax(all_df), np.amax(all_vasc), sep=',', file=outfile)
    elif FLAGS.pool_by_id=='xtrm' :
      s_all.extend(np.column_stack( (extreme_probability(all_mel), extreme_probability(all_nv), extreme_probability(all_bcc), extreme_probability(all_akiec), extreme_probability(all_bkl), extreme_probability(all_df), extreme_probability(all_vasc)) ) )
      print(previous_id, extreme_probability(all_mel), extreme_probability(all_nv), extreme_probability(all_bcc), extreme_probability(all_akiec), extreme_probability(all_bkl), extreme_probability(all_df), extreme_probability(all_vasc), sep=',', file=outfile)
    else :
      raise ValueError('Invalid value for FLAGS.pool_by_id: %s' % FLAGS.pool_by_id)

  for i in xrange(len(image_ids)) :
    if image_ids[i]!=previous_id :
      if previous_id is not None :
        print_result()
      previous_id = image_ids[i]
      all_mel = np.asarray([ predictions_mel[i] ])
      all_nv = np.asarray([ predictions_nv[i] ])
      all_bcc = np.asarray([ predictions_bcc[i] ])
      all_akiec = np.asarray([ predictions_akiec[i] ])
      all_bkl = np.asarray([ predictions_bkl[i] ])
      all_df = np.asarray([ predictions_df[i] ])
      all_vasc = np.asarray([ predictions_vasc[i] ])
    else :
      all_mel = np.concatenate((all_mel, np.asarray([ predictions_mel[i] ])))
      all_nv = np.concatenate((all_nv, np.asarray([ predictions_nv[i] ])))
      all_bcc = np.concatenate((all_bcc, np.asarray([ predictions_bcc[i] ])))
      all_akiec = np.concatenate((all_akiec, np.asarray([ predictions_akiec[i] ])))
      all_bkl = np.concatenate((all_bkl, np.asarray([ predictions_bkl[i] ])))
      all_df = np.concatenate((all_df, np.asarray([ predictions_df[i] ])))
      all_vasc = np.concatenate((all_vasc, np.asarray([ predictions_vasc[i] ])))
  if previous_id is not None :
    print_result()


metfile = open(FLAGS.output_metrics, 'w') if FLAGS.output_metrics else sys.stderr
try :
  accs = []
  aucs = []
  mAPs = []
  accss=[]
  baccs=[]
  p_all=np.array([np.argmax(s) for s in s_all])

  for j, scores_j in [ [0, predictions_mel], [1, predictions_nv],[2, predictions_bcc], [3, predictions_akiec], [4, predictions_bkl], [5, predictions_df], [6, predictions_vasc] ] :
    
    labels_j = (labels_ind == j).astype(np.int)
    
    s_all=np.array(s_all)
    scores_j = s_all[:,j]

    auc = sk.metrics.roc_auc_score(labels_j, scores_j)
    aucs.append(auc)
    print('AUC[%d]: ' % j, auc, file=metfile)

    mAP = sk.metrics.average_precision_score(labels_j, scores_j)
    mAPs.append(mAP)
    print('mAP[%d]: ' % j, mAP, file=metfile)

    pred_j = (p_all == j).astype(np.int)
    
    acc_j = np.mean(np.equal(pred_j, labels_j))
    accss.append(acc_j)
    print('Acc[%d]: ' %j, acc_j, file=metfile)
        
    b_acc_j = np.sum( np.equal(pred_j, 1)& np.equal(labels_j,1) )/np.sum(labels_j)
    baccs.append(b_acc_j)
    print('Bal_Acc[%d]: '%j, b_acc_j, file=metfile)
    
    print('Confusion Matrix:\n', sklearn.metrics.confusion_matrix(labels_j, pred_j), file=metfile)
  
  print('Acc_avg: ', sum(accs) / 7.0, file=metfile)
  print('AUC_avg: ', sum(aucs) / 7.0, file=metfile)
  print('mAP_avg: ', sum(mAPs) / 7.0, file=metfile)
  print('Cat_acc: ', sum(accss) / 7.0, file=metfile)
  print('Bal_acc: ', sum(baccs) / 7.0, file=metfile)
  print('Confusion Matrix:\n', sklearn.metrics.confusion_matrix(labels_ind, p_all), file=metfile)
except ValueError :
  pass

print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)
