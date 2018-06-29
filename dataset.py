#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import n2b2
import numpy, pickle
import configparser, os, nltk, pandas
import glob, string, collections, operator

LABEL2INT = {'not met':0, 'met':1}
INT2LABEL = {0: 'not met', 1: 'met'}
ALPHABET_FILE = 'Model/alphabet.txt'

class DatasetProvider:
  """Comorboditiy data loader"""

  def __init__(self,
               xml_dir,
               cui_dir,
               category,
               use_pickled_alphabet=False,
               alphabet_pickle=None,
               min_token_freq=0,
               use_tokens=False):
    """Index words by frequency in a file"""

    self.xml_dir = xml_dir
    self.cui_dir = cui_dir
    self.category = category
    self.alphabet_pickle = alphabet_pickle
    self.min_token_freq = min_token_freq
    self.use_tokens = use_tokens

    self.token2int = {}

    if use_pickled_alphabet:
      pkl = open(alphabet_pickle, 'rb')
      self.token2int = pickle.load(pkl)

  def load_for_sklearn(self):
    """Load for sklearn training"""

    labels = []    # string labels
    examples = []  # examples as strings

    # document number -> label mapping
    doc2label = n2b2.map_patients_to_labels(
      self.xml_dir,
      self.category)

    for f in os.listdir(self.cui_dir):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.cui_dir, f)
      file_as_string = open(file_path).read()

      string_label = doc2label[doc_id]
      int_label = LABEL2INT[string_label]
      labels.append(int_label)
      examples.append(file_as_string)

    return examples, labels

  def load_for_keras(self, maxlen=float('inf')):
    """Convert examples into lists of indices for keras"""

    labels = []    # int labels
    examples = []  # examples as int sequences

    # document number -> label mapping
    doc2label = n2b2.map_patients_to_labels(
      self.xml_dir,
      self.category)

    # load examples and labels
    for f in os.listdir(self.cui_dir):
      doc_id = f.split('.')[0]
      file_path = os.path.join(self.cui_dir, f)

      if self.use_tokens:
        # read sequence of tokens from file
        file_feat_list = read_tokens(file_path)
      else:
        # read cuis from file as a set
        file_feat_list = read_cuis(file_path)

      example = []
      for token in file_feat_list:
        if token in self.token2int:
          example.append(self.token2int[token])
        else:
          example.append(self.token2int['oov_word'])

      if len(example) > maxlen:
        example = example[0:maxlen]

      string_label = doc2label[doc_id]
      int_label = LABEL2INT[string_label]
      labels.append(int_label)
      examples.append(example)

    return examples, labels

def read_cuis(file_path):
  """Get a set of cuis from a file"""

  file_as_string = open(file_path).read()
  return set(file_as_string.split())

def read_tokens(file_path):
  """Return file as a list of ngrams"""

  text = open(file_path).read().lower()

  tokens = []
  for token in text.split():
    if token.isalpha():
      tokens.append(token)

  return tokens

def load_one_file_for_keras(file_path, alphabet_pickle):
  """Convert single example into a list of integers"""

  file_feat_list = open(file_path).read().split()

  pkl = open(alphabet_pickle, 'rb')
  token2int = pickle.load(pkl)

  example = []
  for token in set(file_feat_list):
    if token in token2int:
      example.append(token2int[token])
    else:
      example.append(token2int['oov_word'])

  return example

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']
  xml_dir = os.path.join(base, cfg.get('data', 'xml_dir'))
  cui_dir = os.path.join(base, cfg.get('data', 'cui_dir'))

  provider = DatasetProvider(
    xml_dir,
    cui_dir,
    'DRUG-ABUSE',
    use_pickled_alphabet=False,
    alphabet_pickle=cfg.get('data', 'alphabet_pickle'))

  x, y = provider.load_for_keras()
  print(x[10])
