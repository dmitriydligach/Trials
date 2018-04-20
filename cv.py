#!/usr/bin/env python

import numpy
numpy.random.seed(0)

import sys
sys.dont_write_bytecode = True
import ConfigParser, os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model
from dataset import DatasetProvider
import n2b2, dataset

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

NUM_FOLDS = 10

def grid_search(x, y):
  """Find best model"""

  param_grid = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced', None]}
  lr = LogisticRegression()
  grid_search = GridSearchCV(
    lr,
    param_grid,
    scoring='f1_micro',
    cv=10)
  grid_search.fit(x, y)

  return grid_search.best_estimator_

def nfold_cv_sparse(category):
  """Run n-fold CV on training set"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_xml_dir = os.path.join(base, cfg.get('data', 'train_xml_dir'))
  train_cui_dir = os.path.join(base, cfg.get('data', 'train_cui_dir'))

  dataset = DatasetProvider(
    train_xml_dir,
    train_cui_dir,
    category,
    use_pickled_alphabet=False,
    alphabet_pickle=cfg.get('model', 'alphabet_pickle'))
  x, y = dataset.load_for_sklearn()

  vectorizer = TfidfVectorizer()
  x = vectorizer.fit_transform(x)

  # print features to file for debugging
  feature_file = open(cfg.get('model', 'feature_list'), 'w')
  for feature in vectorizer.get_feature_names():
    feature_file.write(feature + '\n')

  classifier = grid_search(x, y)
  cv_scores = cross_val_score(
    classifier,
    x,
    y,
    scoring='f1_micro',
    cv=NUM_FOLDS)
  print 'micro f1 (%s) = %.3f' % (category, numpy.mean(cv_scores))

  # train best model on all data and pickle
  classifier.fit(x, y)
  classifier_pickle = 'Model/%s.clf' % category
  vectorizer_pickle = 'Model/%s.vec' % category
  pickle.dump(classifier, open(classifier_pickle, 'wb'))
  pickle.dump(vectorizer, open(vectorizer_pickle, 'wb'))

  return numpy.mean(cv_scores)

def nfold_cv_dense(category):
  """Run n-fold CV on training set"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  train_xml_dir = os.path.join(base, cfg.get('data', 'train_xml_dir'))
  train_cui_dir = os.path.join(base, cfg.get('data', 'train_cui_dir'))

  # load pre-trained model
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(
    inputs=model.input,
    outputs=model.get_layer('HL').output)

  dataset = DatasetProvider(
    train_xml_dir,
    train_cui_dir,
    category,
    use_pickled_alphabet=True,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))
  x, y = dataset.load_for_keras()

  classes = len(set(y))
  maxlen = cfg.getint('data', 'maxlen')
  x = pad_sequences(x, maxlen=maxlen)

  # make training vectors for target task
  x = interm_layer_model.predict(x)

  classifier = grid_search(x, y)
  cv_scores = cross_val_score(
    classifier,
    x,
    y,
    scoring='f1_micro',
    cv=NUM_FOLDS)
  print 'micro f1 (%s) = %.3f' % (category, numpy.mean(cv_scores))

  # train best model on all data and pickle
  classifier.fit(x, y)
  classifier_pickle = 'Model/%s.clf' % category
  pickle.dump(classifier, open(classifier_pickle, 'wb'))

  return numpy.mean(cv_scores)

def nfold_cv_all():
  """Evaluate classifier performance for all 13 conditions"""

  # need train dir to list category names
  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  eval_type = cfg.get('args', 'eval_type')
  train_xml_dir = os.path.join(base, cfg.get('data', 'train_xml_dir'))

  f1s = []
  for category in n2b2.get_category_names(train_xml_dir):

    # weird sklearn bug - skip for now
    if category == 'KETO-1YR':
      continue

    if eval_type == 'sparse':
      # use bag-of-word vectors
      f1 = nfold_cv_sparse(category)
    else:
      # use learned patient vectors
      f1 = nfold_cv_dense(category)

    f1s.append(f1)

  print '--------------------------'
  print 'average micro f1 = %.3f' % numpy.mean(f1s)

if __name__ == "__main__":

  nfold_cv_all()
