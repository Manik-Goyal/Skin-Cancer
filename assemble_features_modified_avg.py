from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import os
import sys
import random
import itertools

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
            'inception' :            { 1 : 'inception.feats1',
                                       2 : 'inception.feats2',
                                       3 : 'inception.feats3', 
                                       4 : 'inception.feats4',
                                       5 : 'inception.feats5',
                                       6 : 'inception.feats6',
                                       7 : 'inception.feats7',
                                       8 : 'inception.feats8',
                                       9 : 'inception.feats9',
                                       0 : 'inception.feats0',},
            'inception_norm' :       { 1 : 'inception_norm.feats1',
                                       2 : 'inception_norm.feats2',
                                       3 : 'inception_norm.feats3',
                                       4 : 'inception_norm.feats4',
                                       5 : 'inception_norm.feats5',
                                       6 : 'inception_norm.feats6',
                                       7 : 'inception_norm.feats7',
                                       8 : 'inception_norm.feats8',
                                       9 : 'inception_norm.feats9',
                                       0 : 'inception_norm.feats0', },
        'inception_resnet' :          { 1 : 'inception_resnet.feats1',
                                       2 : 'inception_resnet.feats2',
                                       3 : 'inception_resnet.feats3',
                                       4 : 'inception_resnet.feats4',
                                       5 : 'inception_resnet.feats5',
                                       6 : 'inception_resnet.feats6',
                                       7 : 'inception_resnet.feats7',
                                       8 : 'inception_resnet.feats8',
                                       9 : 'inception_resnet.feats9',
                                       0 : 'inception_resnet.feats0', },
       'inception_resnet_norm' :     { 1 : 'inception_resnet_norm.feats1',
                                       2 : 'inception_resnet_norm.feats2',
                                       3 : 'inception_resnet_norm.feats3',
                                       4 : 'inception_resnet_norm.feats4',
                                       5 : 'inception_resnet_norm.feats5',
                                       6 : 'inception_resnet_norm.feats6',
                                       7 : 'inception_resnet_norm.feats7',
                                       8 : 'inception_resnet_norm.feats8',
                                       9 : 'inception_resnet_norm.feats9',
                                       0 : 'inception_resnet_norm.feats0', },
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
            features = [ line.strip().split(',') for line in sf] 
            features = { f[0].strip() : [ getfeat(ff.strip()) for ff in f[2:-2] ] for f in features }
            feature_sets.setdefault(feature_key, {})[replica_key] = features
            #print('3', features)
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

feature_order = tuple(('inception', 'inception_norm', 'inception_resnet','inception_resnet_norm'))

feature_combinations = random.sample(list(itertools.product(range(10), repeat=4)), 500)



num_samples = len(image_ids)*len(feature_combinations)

target_file = open(target_file, 'wb')
pickle.dump([ num_samples, FEATURE_LENGTH ], target_file)
for iid in image_ids :
    for comb in feature_combinations :
        feature = np.zeros(7)
        for f, fname in enumerate(feature_order) :
            feature+=(feature_sets[fname][comb[f]][iid])
        
        pickle.dump([ iid, labels[iid], feature/10 ], target_file)


