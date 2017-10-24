import json
import urllib2
import pickle
import operator
from difflib import SequenceMatcher
import csv
import os

def crawlPapers():
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers" 
    fileName = 'additional.txt'
    outputFile = open('failed.txt', 'w')
    files = os.listdir(directory)
    inDictionary = set()
    for f in files:
        inDictionary.add(str(f.split('.')).strip())
    with open(fileName) as f:
        counter = 1
        found = 0
        foundInWeb = 0
        failed = 0
        for line in f:
            line = line.strip()
            if ( line in inDictionary): 
                found = found + 1
            else:
                if line: 
                    try:
                        link = 'http://api.semanticscholar.org/v1/paper/'+line
                        response = urllib2.urlopen(link)
                        data = response.read()
                        filename = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\Dataset\\Papers\\"+str(line) + ".json"
                        file_ = open(filename, 'w')
                        file_.write(data)
                        file_.close()
                        foundInWeb = foundInWeb + 1
                    except urllib2.HTTPError as err:
                        failed = failed + 1
                        outputFile.write(line+'\n')
            print counter, ', found in dir:', found,'found in web:', foundInWeb, 'failed download:', failed
            counter = counter + 1

def main():
    crawlPapers()
    
if __name__ == '__main__':
    main()