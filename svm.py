#!/usr/bin/env python

import numpy
numpy.random.seed(0)

import sys
sys.dont_write_bytecode = True
import ConfigParser, os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model
from dataset import DatasetProvider
import n2b2

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

FEATURE_LIST = 'Model/features.txt'
NUM_FOLDS = 10

def nfold_cv_sparse(category):
  """Run n-fold CV on training set"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  xml_dir = os.path.join(base, cfg.get('data', 'xml_dir'))
  cui_dir = os.path.join(base, cfg.get('data', 'cui_dir'))

  dataset = DatasetProvider(
    xml_dir,
    cui_dir,
    category,
    use_pickled_alphabet=False,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))
  x, y = dataset.load_raw()

  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(x)

  # print features to file for debugging
  feature_file = open(FEATURE_LIST, 'w')
  for feature in vectorizer.get_feature_names():
    feature_file.write(feature + '\n')

  classifier = LinearSVC(class_weight='balanced', C=1)
  cv_scores = cross_val_score(
    classifier,
    tfidf_matrix,
    y,
    scoring='f1_macro',
    cv=NUM_FOLDS)

  print 'macro f1 (%s) = %.3f' % (category, numpy.mean(cv_scores))
  return numpy.mean(cv_scores)

def nfold_cv_sparse_all(eval_type):
  """Evaluate classifier performance for all 13 conditions"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  xml_dir = os.path.join(base, cfg.get('data', 'xml_dir'))

  f1s = []
  for category in n2b2.get_category_names(xml_dir):

    # weird sklearn bug - skip for now
    if category == 'KETO-1YR':
      continue

    if eval_type == 'sparse':
      # use bag-of-word vectors
      f1 = nfold_cv_sparse(category)
    elif evaluation == 'svd':
      # use low dimensional vectors obtained via svd
      p, r, f1 = nfold_cv_svd(category)
    else:
      # use learned patient vectors
      p, r, f1 = nfold_cv_dense(disease, judgement)
    f1s.append(f1)

  print '---------------------------'
  print 'average macro f1 = %.3f' % numpy.mean(f1s)

if __name__ == "__main__":

  nfold_cv_sparse_all('sparse')
