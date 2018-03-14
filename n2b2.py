#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
import xml.etree.ElementTree as et
import os.path, glob, string
from collections import Counter

def generate_dataset_stats(xml_dir):
  """Generate various descriptive stats for the dataset"""

  for category in get_category_names(xml_dir):
    doc2labels = map_patients_to_labels(xml_dir, category)
    counts = Counter(doc2labels.values())
    total = len(doc2labels)
    met = counts['met']
    print 'category:', category.lower()
    print 'positive prevalence: %.1f%%' % \
            (met * 100 / float(total))
    print

def map_patients_to_labels(xml_dir, category):
  """Get a patient number to label mapping"""

  # key: patient num, value: met or not met
  doc2labels = {}

  for xml_file in os.listdir(xml_dir):
    patient_num = xml_file.split('.')[0]
    xml_file_path = os.path.join(xml_dir, xml_file)
    tree = et.parse(xml_file_path)

    for tags in tree.iter(category):
      doc2labels[patient_num] = tags.attrib['met']

  return doc2labels

def get_category_names(xml_dir):
  """Get a list of selection criteria"""

  categories = []

  # take categories from the first file
  for xml_file in os.listdir(xml_dir):
    xml_file_path = os.path.join(xml_dir, xml_file)
    tree = et.parse(xml_file_path)

    for tags in tree.iter('TAGS'):
      for category in tags:
        categories.append(category.tag)
    break

  return categories

def write_notes_to_files(xml_dir, out_dir):
  """Extract notes from xml and write to files"""

  for xml_file in os.listdir(xml_dir):
    xml_file_path = os.path.join(xml_dir, xml_file)
    tree = et.parse(xml_file_path)

    for text in tree.iter('TEXT'):
      narrative = text.text
      printable_narrative = \
        ''.join(c for c in narrative if c in string.printable)
      out_file = xml_file.split('.')[0] + '.txt'
      out_file_path = os.path.join(out_dir, out_file)
      out = open(out_file_path, 'w')
      out.write(printable_narrative + '\n')

if __name__ == "__main__":

  xml_dir = '/Users/Dima/Loyola/Data/Trials/Xml/'
  out_dir = '/Users/Dima/Loyola/Data/Trials/Text/'

  generate_dataset_stats(xml_dir)
