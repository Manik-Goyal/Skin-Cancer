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

parser = argparse.ArgumentParser(prog='train_svm_layer.py', description='Train the SVM decision.')
parser.add_argument('--svm_method', type=str, default='RBF', help='svm method to employ: RBF (default), LINEAR_DUAL, or LINEAR_PRIMAL.')
parser.add_argument('--max_iter_svm', type=int, default=1000, help='maximum number of interations for the linear svm.')
parser.add_argument('--max_iter_hyper', type=int, default=10, help='maximum number of interations for the hyperparameter search.')
parser.add_argument('--jobs', type=int, default=1, help='number of parallel jobs in the hyperparameter search.')
parser.add_argument('--preprocess', type=str, default='PCA', help='train and apply a preprocessor for the data: PCA, PCA_WHITEN, Z_SCORE, NONE.')
parser.add_argument('--input_training', type=str, required=True, help='input file with the training data, in pickle format.')
parser.add_argument('--output_model', type=str, required=True, help='output file to receive the model, in pickle format.')
parser.add_argument('--no_group', default=False, action='store_true', help='do not group samples using id when cross-validating.')
FLAGS = parser.parse_args()

valid_svm_methods = [ 'RBF', 'LINEAR_DUAL', 'LINEAR_PRIMAL' ]
if not FLAGS.svm_method in valid_svm_methods :
    print('--svm_method must be one of ', ', '.join(valid_svm_methods), file=sys.stderr)
    sys.exit(1)
SVM_LINEAR = FLAGS.svm_method == 'LINEAR_DUAL' or FLAGS.svm_method == 'LINEAR_PRIMAL'
SVM_DUAL = FLAGS.svm_method == 'LINEAR_DUAL'

SVM_MAX_ITER = FLAGS.max_iter_svm
HYPER_MAX_ITER = FLAGS.max_iter_hyper
HYPER_JOBS = FLAGS.jobs

valid_preprocesses = [ 'PCA', 'PCA_WHITEN', 'Z_SCORE', 'NONE' ]
if not FLAGS.preprocess in valid_preprocesses :
    print('--preprocess must be one of ', ' '.join(valid_preprocesses), file=sys.stderr)
    sys.exit(1)

first = start = su.print_and_time('Reading training data...', file=sys.stderr)
ids, labels, features = su.read_pickled_data(FLAGS.input_training)
start = su.print_and_time('', past=start, file=sys.stderr)


num_samples = len(ids)
min_gamma   = np.floor(np.log2(1.0/num_samples)) - 4.0
max_gamma   = min(3.0, min_gamma+32.0)
scale_gamma = max_gamma-min_gamma
print('\tSamples: ', num_samples, file=sys.stderr)
if not SVM_LINEAR :
    print('\tGamma: ', min_gamma, min_gamma+scale_gamma, file=sys.stderr)

start = su.print_and_time('Training preprocessor...', file=sys.stderr)

if FLAGS.preprocess == 'PCA' :
    preprocessor = sk.decomposition.PCA(copy=False, whiten=False)
elif FLAGS.preprocess == 'PCA_WHITEN' :
    preprocessor = sk.decomposition.PCA(copy=False, whiten=True)
elif FLAGS.preprocess == 'Z_SCORE' :
    preprocessor = sk.preprocessing.StandardScaler(copy=False)
elif FLAGS.preprocess == 'NONE' :
    # func=None implies identity function
    preprocessor = sk.preprocessing.FunctionTransformer(func=None, inverse_func=None, validate=False,
        accept_sparse=False, pass_y=False, kw_args=None, inv_kw_args=None)
else :
    assert False, '(bug) Invalid value for FLAGS.preprocess: %s' % FLAGS.preprocess
features = preprocessor.fit_transform(features)

group_msg = 'ungrouped' if FLAGS.no_group else 'grouped'

start = su.print_and_time('====================\nTraining mel classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_mel = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_mel.fit(features, (labels==0).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_mel.best_params_, file=sys.stderr)
print('...', classifier_mel.best_params_, end='', file=sys.stderr)

start = su.print_and_time('====================\nTraining nv classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_nv = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_nv.fit(features, (labels==1).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_nv.best_params_, file=sys.stderr)
print('...', classifier_nv.best_params_, end='', file=sys.stderr)

start = su.print_and_time('====================\nTraining bcc classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_bcc = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_bcc.fit(features, (labels==2).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_bcc.best_params_, file=sys.stderr)
print('...', classifier_bcc.best_params_, end='', file=sys.stderr)


start = su.print_and_time('====================\nTraining akiec classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_akiec = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_akiec.fit(features, (labels==3).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_akiec.best_params_, file=sys.stderr)
print('...', classifier_akiec.best_params_, end='', file=sys.stderr)


start = su.print_and_time('====================\nTraining bkl classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_bkl = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_bkl.fit(features, (labels==4).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_bkl.best_params_, file=sys.stderr)
print('...', classifier_bkl.best_params_, end='', file=sys.stderr)


start = su.print_and_time('====================\nTraining df classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_df = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_df.fit(features, (labels==5).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_df.best_params_, file=sys.stderr)
print('...', classifier_df.best_params_, end='', file=sys.stderr)


start = su.print_and_time('====================\nTraining vasc classifier (%s)...\n' % group_msg, past=start, file=sys.stderr)
classifier, tuning = su.new_classifier(linear=SVM_LINEAR, dual=SVM_DUAL, max_iter=SVM_MAX_ITER, min_gamma=min_gamma, scale_gamma=scale_gamma)
classifier_vasc = su.hyperoptimizer(classifier, tuning, max_iter=HYPER_MAX_ITER, n_jobs=HYPER_JOBS, group=not FLAGS.no_group)
classifier_vasc.fit(features, (labels==6).astype(np.int), groups=None if FLAGS.no_group else ids)
print('Best params:', classifier_vasc.best_params_, file=sys.stderr)
print('...', classifier_vasc.best_params_, end='', file=sys.stderr)

start = su.print_and_time('====================\nWriting model...', past=start, file=sys.stderr)
model_file = open(FLAGS.output_model, 'wb')
pickle.dump(preprocessor, model_file)
pickle.dump(classifier_mel, model_file)
pickle.dump(classifier_nv, model_file)
pickle.dump(classifier_bcc, model_file)
pickle.dump(classifier_akiec, model_file)
pickle.dump(classifier_bkl, model_file)
pickle.dump(classifier_df, model_file)
pickle.dump(classifier_vasc, model_file)
pickle.dump(FLAGS, model_file)
model_file.close()

print('\n Total time ', end='', file=sys.stderr)
_ = su.print_and_time('Done!\n', past=first, file=sys.stderr)
