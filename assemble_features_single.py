from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import os
import sys

import numpy as np
import scipy as sp

if sys.argv[1]=='LOGITS' :
    use_logits = True
    all_combinations = False
elif sys.argv[1]=='ALL_LOGITS' :
    use_logits = True
    all_combinations = True
elif sys.argv[1]=='PROBS' :
    use_logits = False
    all_combinations = False
elif sys.argv[1]=='ALL_PROBS' :
    use_logits = False
    all_combinations = True
else :
    print('ERROR: First argument must be [ALL_]LOGITS or [ALL_]PROBS, instead got %s' % sys.argv, file=sys.stderr)
    sys.exit(1)

source_folder = sys.argv[2]
target_file = sys.argv[3]
labels_file = sys.argv[4] if len(sys.argv)>4 else None

feature_files = \
    {
            'inception_avg' :        { 1 : 'inception_1.feats',
                                       2 : 'inception_2.feats',
                                       3 : 'inception_3.feats', },
            'inception_norm_avg' :   { 1 : 'inception_norm_1.feats',
                                       2 : 'inception_norm_2.feats',
                                       3 : 'inception_norm_3.feats', },
        'inception_resnet_avg' :     { 1 : 'inception_resnet_1.feats',
                                       2 : 'inception_resnet_2.feats',
                                       3 : 'inception_resnet_3.feats', },
    }



def bits_from_probstr(probstr) :
    prob = float(probstr)
    prob = max(bits_from_probstr.SCEPTICISM_THRESHOLD, prob)
    prob = min(1.0-bits_from_probstr.SCEPTICISM_THRESHOLD, prob)
    odds = prob/(1.0-prob)
    bits = np.log2(odds)
    return bits
# This corresponds to doubting the model can do better than one error in one million
bits_from_probstr.SCEPTICISM_THRESHOLD=0.00001

if use_logits :
    getfeat = bits_from_probstr
else :
    getfeat = float

FEATURE_LENGTH = 7

image_ids = None
feature_sets = {}
for feature_key in feature_files.keys() :
    for replica_key in feature_files[feature_key].keys() :
        with open(os.path.join(source_folder, feature_files[feature_key][replica_key]), 'r') as sf:
            next(sf)
            features = [ line.strip().split(',') for line in sf] 
            features = { f[0].strip() : [ getfeat(ff.strip()) for ff in f[2:-1] ] for f in features }
            feature_sets.setdefault(feature_key, {})[replica_key] = features
            feature_ids = sorted(features.keys())
            if image_ids is None :
                image_ids = feature_ids
            else :
                if feature_ids != image_ids  :
                    print('WARNING: mismatched image id list in %s' % feature_files[feature_key][replica_key], file=sys.stderr)
if not labels_file is None :
    labels = [ line.strip().split(',') for line in open(labels_file, 'r') ]
    labels = { label[0].strip() : [i for i, e in enumerate(label[1:]) if e != '0.0'][0]
        for label in labels[1:] }
else :
    print('WARNING: no label file informed --- assuming this is a test split', file=sys.stderr)
    labels = { iid : 0 for iid in image_ids }

feature_order = tuple(('inception_avg', 'inception_norm_avg', 'inception_resnet_avg'))

feature_combinations = [ [1]*3 ]
#feature_combinations = [(1, 2, 3), (1, 1, 2), (1, 3, 1), (2, 1, 1), (3, 3, 1), (3, 2, 2), (3, 1, 2), (2, 1, 2), (2, 3, 1), (2, 2, 3), (1, 2, 2), (3, 1, 1), (2, 3, 2), (1, 2, 1), (1, 1, 3), (2, 1, 3), (3, 2, 3), (2, 3, 3), (3, 3, 2), (2, 2, 1), (2, 2, 2), (1, 3, 3), (3, 1, 3), (1, 3, 2), (3, 3, 3), (3, 2, 1), (1, 1, 1)]



num_samples = len(image_ids)*len(feature_combinations)

target_file = open(target_file, 'wb')
pickle.dump([ num_samples, len(feature_order)*FEATURE_LENGTH ], target_file)
for iid in image_ids :
    for comb in feature_combinations :
        feature = []
        for f, fname in enumerate(feature_order) :
            feature.extend(feature_sets[fname][comb[f]][iid])
        pickle.dump([ iid, labels[iid], feature ], target_file)


