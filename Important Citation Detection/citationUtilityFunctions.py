import csv
import os
import json
from difflib import SequenceMatcher
import pickle
import urllib2
import shutil

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def checkCitedBy():
    SS = set()
    unique = set()
    fileName = 'similarityUnique.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[0].strip()
            SS.add(paper_id)
    print len(SS)
    count = 0
    with open('citations.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
#             paperid = elements[0]
#             if paperid not in unique:
#                 unique.add(paperid)
#                 count = count + 1
            paperid = elements[1]
            if paperid not in unique:
                unique.add(paperid)
            count = count + 1
    print count
    count = 0
    outputFile = open('additional.txt', 'w')
    for paperid in unique:
        if paperid  not in SS:    
            outputFile.write(paperid + '\n')
            count = count + 1
    outputFile.close()        
    print count, len(unique)
    
def getCitations():
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers" 
    files = os.listdir(directory)
    inDictionary = set()
    for f in files:
        inDictionary.add(str(f.split('.')[0]).strip())
        
    papers = set()
    with open('PaperList.txt', 'rb') as f:
        for line in f:
            elements = line.strip().split('\t')
            paper_id = elements[1]
            papers.add(paper_id)
    print len(papers)
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers" 
    outputFile = open('citations.csv', 'w')
    for paper in papers:
        if paper in inDictionary:
            x_file = open(os.path.join(directory, paper + '.json'), "r")
            data = json.load(x_file)
            citations = data['citations']
            for citation in citations:
                citationId = citation['paperId']
                labelinfo = citation['isInfluential']
                label = 0
                if (labelinfo == True):
                    label = 1
                outputFile.write(paper + ',' + citationId + ',' + str(label) + '\n')
        else:
            print paper
                

def updateCitations():
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers" 
    files = os.listdir(directory)
    SS = set()
    for f in files:
        SS.add(str(f.split('.')[0]).strip())
        
    outputFile = open('citations_updated_2.csv', 'w')
    with open('citations_updated.csv', 'rb') as f:
        for line in f:
            elements = line.strip().split(',')
            paperid2 = elements[1]
            if paperid2 in SS:
                outputFile.write(line.strip() + '\n')
    outputFile.close()
    
def compareCitations():
    outputFile = open('citations_updated_3.csv', 'w')
    count = 1
    with open('citations.csv', 'rb') as f:
        for line in f:
            print 'paper:', count
            elements = line.strip().split(',')
            paper = elements[0].strip()
            citedBy = elements[1].strip()
            label = int(elements[2].strip())
            with open('annotatedUpdated.csv', 'rb') as f1:
                f1.readline()
                label2 = -1
                for line1 in f1:   
                    elements = line1.strip().split(',')
                    paper2 = elements[1].strip()
                    citedBy2 = elements[3].strip()
                    if((paper == paper2) and (citedBy == citedBy2)):
                        label2 = int(elements[4])
                        break
            outputFile.write(paper + ',' + citedBy + ',' + str(label) + ',' + str(label2)+'\n')
            count = count + 1
    outputFile.close()
    
def countUnique():
    SS = set()
    with open('annotatedUpdated.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:   
            elements = line1.strip().split(',')
            paper2 = elements[1].strip()
            SS.add(paper2) 
    SS1 = set()
    with open('citations.csv', 'rb') as f1:
        for line1 in f1:   
            elements = line1.strip().split(',')
            paper2 = elements[0].strip()
            SS1.add(paper2) 
    SS2 = set()
    with open('not_found.txt', 'rb') as f1:
        for line1 in f1:   
            elements = line1.strip().split('\t')
            paper2 = elements[0].strip()
            SS2.add(paper2)       
            
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers" 
    files = os.listdir(directory)
    SS3 = set()
    for f in files:
        SS3.add(str(f.split('.')[0]).strip())
    
    for paper in SS:
        if paper not in SS3:
            print paper
            
    print len(SS),len(SS1),len(SS2)

def uniquePapers():
    AS = set()
    SS = set()
    with open('annotatedUpdated.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:   
            elements = line1.strip().split(',')
            SS.add(elements[1].strip()) 
            SS.add(elements[3].strip())
            AS.add(elements[0].strip()) 
            AS.add(elements[2].strip())
    print len(AS),len(SS)

def mapping():
    AS = set()
    SS = set()
    maps = dict()
    with open('annotatedUpdated.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:   
            elements = line1.strip().split(',')
            paperACL = elements[0].strip()
            paperSS = elements[1].strip()
            citedByACL = elements[2].strip()
            citedBySS = elements[3].strip()
            if paperACL in maps:
                if maps[paperACL] != paperSS:
                    print '1', paperACL, paperSS
            else:
                maps[paperACL] = paperSS
            if citedByACL in maps:
                if maps[citedByACL] != citedBySS:
                    print '2', citedByACL, maps[citedByACL], citedBySS
            else:
                maps[citedByACL] = citedBySS
            SS.add(elements[1].strip()) 
            SS.add(elements[3].strip())
            AS.add(elements[0].strip()) 
            AS.add(elements[2].strip())
            
    print len(AS),len(SS)

def findMapping():
    maps = dict()
    outputFile = open('mapping.txt', 'w')
    with open('annotatedUpdated.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:   
            elements = line1.strip().split(',')
            paperACL = elements[0].strip()
            paperSS = elements[1].strip()
            citedByACL = elements[2].strip()
            citedBySS = elements[3].strip()
            if paperSS in maps:
                if maps[paperSS] != paperACL:
                    print '1', paperSS, maps[paperSS], paperACL
            else:
                maps[paperSS] = paperACL
            if citedBySS in maps:
                if maps[citedBySS] != citedByACL:
                    print '2', citedBySS, maps[citedBySS], citedByACL
                    print line1
            else:
                maps[citedBySS] = citedByACL
    print len(maps) 
    
def compareCitationsFinal():
    outputFile = open('citations_updated_4.csv', 'w')
    count = 1
    with open('annotatedUpdated.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:
            print 'citation:', count
            flag = False
            elements = line1.strip().split(',')
            paper2 = elements[1].strip()
            citedBy2 = elements[3].strip()
            label2 = int(elements[4])
            with open('citations.csv', 'rb') as f:
                for line in f:
                    elements = line.strip().split(',')
                    paper = elements[0].strip()
                    citedBy = elements[1].strip()
                    label = int(elements[2].strip())
                    if((paper == paper2) and (citedBy == citedBy2)):
                        flag = True
                        break
            if flag:
                outputFile.write(paper + ',' + citedBy + ',' + str(label) + ',' + str(label2)+'\n')
            count = count + 1
    outputFile.close()  

def findMatching():
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers"
    ACL = pickle.load(open("ACL.p", "rb")) 
    files = os.listdir(directory)
    SS = set()
    for f in files:
        SS.add(str(f.split('.')[0]).strip())
    maps = dict()    
    with open('annotatedUpdated.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:   
            elements = line1.strip().split(',')
            paperACL = elements[0].strip()
            paperSS = elements[1].strip()
            citedByACL = elements[2].strip()
            citedBySS = elements[3].strip()
            if paperACL not in maps:
                maps[paperACL] = paperSS
            if citedByACL not in maps:
                maps[citedByACL] = citedBySS
    outputFile = open('similarityCheck.txt', 'w')
    for key,val in maps.iteritems():
        paperACL = key      
        paperSS = val       
        title_acl = str(ACL[paperACL])        
        x_file = open(os.path.join(directory, paperSS + '.json'), "r")
        data = json.load(x_file)
        title_ss = str(data['title'])
        ratio = similar(title_acl, title_ss)
        outputFile.write(paperSS + '\t' + paperACL + '\t' + title_ss + '\t' + title_acl + '\t' + str(ratio) + '\n')    
    outputFile.close()

def findRestCitations():
    SS = set()
    with open('not_found.txt', 'rb') as f:
        for line in f:
            elements = line.strip().split('\t')
            paper = elements[0]
            SS.add(paper)
    print len(SS)
    
def checkData():
    SS1 = set()
    with open('citations.csv', 'rb') as f1:
        for line1 in f1:   
            elements = line1.strip().split(',')
            paper2 = elements[0].strip()
            SS1.add(paper2)
            paper2 = elements[1].strip()
            SS1.add(paper2) 
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers" 
    files = os.listdir(directory)
    inDictionary = set()
    for f in files:
        inDictionary.add(str(f.split('.')[0]).strip())
    count = 0
    notFound = 0
    
    for paper in SS1:
        if paper in inDictionary:
            count  = count + 1
        else:
            notFound = notFound + 1
#             try:
#                 link = 'http://api.semanticscholar.org/v1/paper/'+paper
#                 response = urllib2.urlopen(link)
#                 data = response.read()
#                 filename = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers\\"+str(paper) + ".json"
#                 file_ = open(filename, 'w')
#                 file_.write(data)
#                 file_.close()
#                 print 'found:',paper
#             except urllib2.HTTPError as err:
#                 continue
    print len(SS1), count, notFound  

def copyFiles():
    SS1 = set()
    with open('citations.csv', 'rb') as f1:
        for line1 in f1:   
            elements = line1.strip().split(',')
            paper2 = elements[0].strip()
            SS1.add(paper2)
            paper2 = elements[1].strip()
            SS1.add(paper2)
    
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers" 
    dest = "Papers\\"
    files = os.listdir(directory)
    inDictionary = set()
    for f in files:
        inDictionary.add(str(f.split('.')[0]).strip())
    count = 1
    for paper in SS1:
        if paper in inDictionary:
            filename = directory+"\\"+str(paper) + ".json"
            if (os.path.isfile(filename)):
                shutil.copy(filename, dest)
                print count
                count = count + 1    
def countUniquePapers():
    SS = set()
    with open('citation_dataset.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:   
            elements = line1.strip().split(',')
            paper2 = elements[1].strip()
            SS.add(paper2) 
            paper2 = elements[0].strip()
            SS.add(paper2) 
    SS1 = set()
    with open('citation_dataset_small.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:   
            elements = line1.strip().split(',')
            paper2 = elements[0].strip()
            SS1.add(paper2)  
            paper2 = elements[1].strip()
            SS1.add(paper2) 
    
    directory = "Papers" 
    files = os.listdir(directory)
    SS3 = set()
    for f in files:
        SS3.add(str(f.split('.')[0]).strip())
    
    small = 0
    large = 0
    for paper in SS:
        if paper not in SS3:
            large = large + 1
            
    for paper in SS1:
        if paper not in SS3:
            print paper
            small = small + 1
            
    print large, small        

def deleteCitations():
    directory = "Papers" 
    files = os.listdir(directory)
    SS3 = set()
    for f in files:
        SS3.add(str(f.split('.')[0]).strip())
    
    SS = set()
    with open('citation_dataset.csv', 'rb') as f1:
        f1.readline()
        for line1 in f1:   
            elements = line1.strip().split(',')
            paper2 = elements[1].strip()
            SS.add(paper2) 
            paper2 = elements[0].strip()
            SS.add(paper2) 
    marker = set()
    for paper in SS:
        if paper not in SS3:
            marker.add(paper)
    
    outputFile = open('citations_dataset_updated.csv', 'w')        
    with open('citation_dataset.csv', 'rb') as f1:
        outputFile.write(f1.readline().strip()+'\n')
        for line1 in f1:   
            elements = line1.strip().split(',')
            paper1 = elements[0].strip()
            paper2 = elements[1].strip()
            if (paper1 in marker or paper2 in marker):
                continue
            else:
                outputFile.write(line1.strip()+'\n')
    outputFile.close()
    
def main():
    deleteCitations()
    
if __name__ == '__main__':
    main()
