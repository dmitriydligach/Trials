#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')
import n2b2
import numpy, pickle
import ConfigParser, os, nltk, pandas
import glob, string, collections, operator

# can be used to turn this into a binary task
LABEL2INT = {'not met':0, 'met':1}
# file to log alphabet entries for debugging
ALPHABET_FILE = 'Model/alphabet.txt'

class DatasetProvider:
  """Comorboditiy data loader"""

  def __init__(self,
               xml_dir,
               cui_dir,
               category,
               use_pickled_alphabet=False,
               alphabet_pickle=None,
               min_token_freq=0):
    """Index words by frequency in a file"""

    self.xml_dir = xml_dir
    self.cui_dir = cui_dir
    self.category = category
    self.alphabet_pickle = alphabet_pickle
    self.min_token_freq = min_token_freq

    self.token2int = {}

    # when training, make alphabet and pickle it
    # when testing, load it from pickle
    if use_pickled_alphabet:
      print 'reading alphabet from', alphabet_pickle
      pkl = open(alphabet_pickle, 'rb')
      self.token2int = pickle.load(pkl)
    else:
      self.make_token_alphabet()

  def make_token_alphabet(self):
    """Map tokens (CUIs) to integers"""

    # count tokens in the entire corpus
    token_counts = collections.Counter()

    for f in os.listdir(self.cui_dir):
      file_path = os.path.join(self.cui_dir, f)
      file_feat_list = open(file_path).read().split()
      token_counts.update(file_feat_list)

    # now make alphabet (high freq tokens first)
    index = 1
    self.token2int['oov_word'] = 0
    outfile = open(ALPHABET_FILE, 'w')
    for token, count in token_counts.most_common():
      if count > self.min_token_freq:
        outfile.write('%s|%s\n' % (token, count))
        self.token2int[token] = index
        index = index + 1

    # pickle alphabet
    pickle_file = open(self.alphabet_pickle, 'wb')
    pickle.dump(self.token2int, pickle_file)

  def load_raw(self):
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
      file_feat_list = open(file_path).read()

      string_label = doc2label[doc_id]
      int_label = LABEL2INT[string_label]
      labels.append(int_label)
      examples.append(file_feat_list)

    return examples, labels

if __name__ == "__main__":

  cfg = ConfigParser.ConfigParser()
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

  x, y = provider.load_raw()
  print sum(y)
