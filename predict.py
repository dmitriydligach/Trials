#!/usr/bin/env python

import numpy
numpy.random.seed(0)

import sys
sys.dont_write_bytecode = True
import ConfigParser, os, pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Model
from dataset import DatasetProvider
import n2b2, dataset

def predict_sparse(train_xml_dir):
  """Make predictions and add them to XML"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  test_xml_dir = os.path.join(base, cfg.get('data', 'test_xml_dir'))
  test_cui_dir = os.path.join(base, cfg.get('data', 'test_cui_dir'))

  for f in os.listdir(test_cui_dir):

    file_path = os.path.join(test_cui_dir, f)
    file_as_string = open(file_path).read()

    category2label = {} # key: selection criterion, value: prediction
    for category in n2b2.get_category_names(train_xml_dir):

      # could not train model; always predict 'not met'
      if category == 'KETO-1YR':
        category2label[category] = 'not met'
        print 'file: %s, crit: %s, label: %s' % (f, category, 'not met')
        continue

      vectorizer_pickle = 'Model/%s.vec' % category
      vectorizer = pickle.load(open(vectorizer_pickle, 'rb'))
      x = vectorizer.transform([file_as_string])

      classifier_pickle = 'Model/%s.clf' % category
      classifier = pickle.load(open(classifier_pickle, 'rb'))
      prediction = classifier.predict(x)
      label = dataset.INT2LABEL[prediction[0]]

      category2label[category] = label
      print 'file: %s, crit: %s, label: %s' % (f, category, label)

    xml_file_name = f.split('.')[0] + '.xml'
    xml_file_path = os.path.join(test_xml_dir, xml_file_name)
    n2b2.write_category_labels(xml_file_path, category2label)
    print

def predict_dense(train_xml_dir):
  """Make predictions and add to XML"""

  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  alphabet_pickle=cfg.get('data', 'alphabet_pickle')
  test_xml_dir = os.path.join(base, cfg.get('data', 'test_xml_dir'))
  test_cui_dir = os.path.join(base, cfg.get('data', 'test_cui_dir'))

  # load pre-trained model
  model = load_model(cfg.get('data', 'model_file'))
  interm_layer_model = Model(
    inputs=model.input,
    outputs=model.get_layer('HL').output)

  for f in os.listdir(test_cui_dir):
    file_path = os.path.join(test_cui_dir, f)
    int_seq = dataset.load_one_file_for_keras(file_path, alphabet_pickle)

    category2label = {}
    # run this file through the model for each selection criterion
    for category in n2b2.get_category_names(train_xml_dir):

      # could not train model; always predict 'not met'
      if category == 'KETO-1YR':
        category2label[category] = 'not met'
        print 'file: %s, crit: %s, label: %s' % (f, category, 'not met')
        continue

      maxlen = cfg.getint('data', 'maxlen')
      x = pad_sequences([int_seq], maxlen=maxlen)
      x = interm_layer_model.predict(x)

      classifier_pickle = 'Model/%s.clf' % category
      classifier = pickle.load(open(classifier_pickle, 'rb'))
      prediction = classifier.predict(x)
      label = dataset.INT2LABEL[prediction[0]]

      category2label[category] = label
      print 'file: %s, crit: %s, label: %s' % (f, category, label)

    xml_file_name = f.split('.')[0] + '.xml'
    xml_file_path = os.path.join(test_xml_dir, xml_file_name)
    n2b2.write_category_labels(xml_file_path, category2label)
    print

def predict_all():
  """Evaluate classifier performance for all 13 conditions"""

  # need train dir to list category names
  cfg = ConfigParser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  eval_type = cfg.get('args', 'eval_type')
  train_xml_dir = os.path.join(base, cfg.get('data', 'train_xml_dir'))

  if eval_type == 'sparse':
    predict_sparse(train_xml_dir)
  else:
    predict_dense(train_xml_dir)

if __name__ == "__main__":

  # nfold_cv_sparse_all()
  predict_all()
