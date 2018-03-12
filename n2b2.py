#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
import xml.etree.ElementTree as et
import os.path, glob, string

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

  write_notes_to_files(xml_dir, out_dir)
