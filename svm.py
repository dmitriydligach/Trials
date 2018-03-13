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
  x, y = dataset.load_for_sklearn()

  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(x)

  # print features to file for debugging
  feature_file = open(FEATURE_LIST, 'w')
  for feature in vectorizer.get_feature_names():
    feature_file.write(feature + '\n')

  classifier = LinearSVC(loss='hinge', class_weight='balanced', C=1)
  cv_scores = cross_val_score(
    classifier,
    tfidf_matrix,
    y,
    scoring='f1_macro',
    cv=NUM_FOLDS)

  print 'macro f1 (%s) = %.3f' % (category, numpy.mean(cv_scores))
  return numpy.mean(cv_scores)

def nfold_cv_dense(category):
  """Run n-fold CV on training set"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  xml_dir = os.path.join(base, cfg.get('data', 'xml_dir'))
  cui_dir = os.path.join(base, cfg.get('data', 'cui_dir'))

  # load pre-trained model
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(
    inputs=model.input,
    outputs=model.get_layer('HL').output)

  dataset = DatasetProvider(
    xml_dir,
    cui_dir,
    category,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))
  x, y = dataset.load_for_keras()

  classes = len(set(y))
  print 'unique labels in train:', classes
  maxlen = cfg.getint('data', 'maxlen')
  x = pad_sequences(x, maxlen=maxlen)

  # make training vectors for target task
  print 'original x_train shape:', x.shape
  x = interm_layer_model.predict(x)
  print 'new x_train shape:', x.shape

  classifier = LinearSVC(loss='hinge', class_weight='balanced', C=1)
  cv_scores = cross_val_score(
    classifier,
    x,
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
    elif eval_type == 'svd':
      # use low dimensional vectors obtained via svd
      f1 = nfold_cv_svd(category)
    else:
      # use learned patient vectors
      f1 = nfold_cv_dense(category)
    f1s.append(f1)

  print '---------------------------'
  print 'average macro f1 = %.3f' % numpy.mean(f1s)

if __name__ == "__main__":

  nfold_cv_sparse_all('dense')
