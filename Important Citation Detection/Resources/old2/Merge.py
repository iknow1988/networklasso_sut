import csv
import os
import json
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def initialMerge():
    hashmap = dict()
    fileName = 'similarity.txt'
    
    with open(fileName) as f:
        for line in f:
            SS_paper_id = line.split('\t')[0].strip()
            ACL_paper_id = line.split('\t')[1].strip()
            if ACL_paper_id not in hashmap:
                hashmap[ACL_paper_id] = SS_paper_id

    outputFile = open('annotatedUpdated.csv', 'w')
    with open('annotated.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            paper = str(elements[1])
            cited_by = str(elements[2])
            followup = str(elements[3])
            paper_SS = hashmap[paper]
            cited_by_SS = hashmap[cited_by]
            outputFile.write(paper + ',' + paper_SS + ',' + cited_by + ',' + cited_by_SS + ',' + followup + '\n')            
    
    outputFile.close()

def paperInformation():
    papers = set()
    hashmap = dict()
    fileName = 'similarity.txt'
    
    with open(fileName) as f:
        for line in f:
            SS_paper_id = line.split('\t')[0].strip()
            ACL_paper_id = line.split('\t')[1].strip()
            if ACL_paper_id not in hashmap:
                hashmap[ACL_paper_id] = SS_paper_id

    outputFile = open('PaperList.txt', 'w')
    with open('annotated.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            paper = str(elements[1])
            papers.add(paper)
            
    for paper in papers:
        outputFile.write(paper + '\t' + hashmap[paper] + '\n')
    print len(papers)
    
def getImportance():
    important = 0
    unimportant = 0
    with open('annotatedUpdated.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            followup = int(elements[4])
            if (followup == 0 or followup == 1):
                unimportant = unimportant + 1
            if (followup == 2 or followup == 3):
                important = important + 1
    
    print unimportant, important
    
def getCitations():
    papers = set()
    with open('PaperList.txt', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split('\t')
            paper_id = elements[1]
            papers.add(paper_id)
    directory = "C:\\Users\\kadnan\\Desktop\\Dataset\\Semantic Scholar\\Papers" 
    outputFile = open('citations.csv', 'w')
    for paper in papers:
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

def checkCitedBy():
    SS = set()
    unique = set()
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[0].strip()
            SS.add(paper_id)
    
    count = 0
    with open('annotatedUpdated.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            paperid = elements[3]
            unique.add(paperid)
            count = count + 1
    print count
    count = 0
#     outputFile = open('additional.txt', 'w')
#     for paperid in unique:
#         if paperid  not in SS:
#             outputFile.write(paperid+'\n')
#         else:
#             count = count + 1
    print count, len(unique)
    
def checkUnique():
    SS = set()
    ACL = set()
    fileName = 'similarityUnique.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[0].strip()
            SS.add(paper_id)
            paper_id = line.split('\t')[1].strip()
            ACL.add(paper_id)
    print len(SS), len(ACL)
    
def checkCitations():
    SS = set()
    unique = set()
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[0].strip()
            SS.add(paper_id)
    
    count = 0
    with open('citations.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            paperid = elements[0]
            unique.add(paperid)
            count = count + 1
    print count
    count = 0
    outputFile = open('additional.txt', 'w')
    for paperid in unique:
        if paperid  not in SS:
            outputFile.write(paperid + '\n')
        else:
            count = count + 1
    print count, len(unique)

def checkduplicate():
    fileName = 'duplicateIds.txt'
    outputFile = open('mismatch.txt', 'w')
    with open(fileName) as f:
        for line in f:
            elements = line.strip().split('\t')
            SSpaper_id = elements[0]
            ACLpaper_id = elements[1]
            SSfile = 'SS.txt'
            ACLFile = 'ACL.txt'
            with open(ACLFile) as f:
                for line in f:
                    aclid = line.strip().split('\t')[0]
                    if (aclid == ACLpaper_id):
                        title_acl = line.strip().split('\t')[2]
#                         outputFile.write(line.strip().split('\t')[2]+'\t')
                        break
            with open(SSfile) as f:
                for line in f:
                    data = json.loads(line)
                    ssid = str(data['id'])
                    if (ssid == SSpaper_id):
                        title_ss = str(data['title'])
#                         outputFile.write(data['title']+'\n')
                        break
            ratio = similar(title_acl, title_ss)
            outputFile.write(SSpaper_id + '\t' + title_ss + '\t' + ACLpaper_id + '\t' + title_acl + '\t' + str(ratio) + '\n')
    outputFile.close()
    
def checkduplicateSimilarity():
    SS = set()
    fileName = 'duplicate.txt'
    with open(fileName) as f:
        for line in f:
            elements = line.strip().split('        ')
            SSpaper_id = elements[0]
            SS.add(SSpaper_id)
            
    outputFile = open('duplicateIds.txt', 'w')
    count = 1
    
    for SSpaper_id in SS:
        print count
        with open('similarity.txt') as f1:
            for line in f1:
                elements = line.strip().split('\t')
                ss = elements[0].strip()
                if(ss == SSpaper_id):
                    outputFile.write(line)
        count = count + 1
    outputFile.close()
              
def perfectMatch():
    fileName = 'duplicateIds.txt'
    outputFile = open('perfectMatch.txt', 'w')
    with open(fileName) as f:
        for line in f:
            elements = line.strip().split('\t')
            SSpaper_id = elements[0]
            ACLpaper_id = elements[1]
            SSfile = 'SS.txt'
            ACLFile = 'ACL.txt'
            ACL_year = 0
            SS_year = 0
            with open(ACLFile) as f:
                for line in f:
                    aclid = line.strip().split('\t')[0]
                    ACL_year = int(line.strip().split('\t')[4])
                    if (aclid == ACLpaper_id):
                        title_acl = line.strip().split('\t')[2]
#                         outputFile.write(line.strip().split('\t')[2]+'\t')
                        break
            with open(SSfile) as f:
                for line in f:
                    data = json.loads(line)
                    ssid = str(data['id'])
                    SS_year = int(data['year'])
                    if (ssid == SSpaper_id):
                        title_ss = str(data['title'])
#                         outputFile.write(data['title']+'\n')
                        break
            ratio = similar(title_acl, title_ss)
            if (ACL_year==SS_year):
                outputFile.write(SSpaper_id + '\t' + title_ss + '\t' + ACLpaper_id + '\t' + title_acl + '\t' + str(ratio) + '\n')
    outputFile.close()  

def usePerfectMatch():
    hashmap = dict()
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            SS_paper_id = line.split('\t')[0].strip()
            ACL_paper_id = line.split('\t')[1].strip()
            if ACL_paper_id not in hashmap:
                hashmap[ACL_paper_id] = SS_paper_id
    print len(hashmap)
    pefectMatchMap = dict()          
    fileName = 'perfectMatch.txt'
    with open(fileName) as f:
        for line in f:
            SS_paper_id = line.split('\t')[0].strip()
            ACL_paper_id = line.split('\t')[2].strip()
            if SS_paper_id not in pefectMatchMap:
                pefectMatchMap[SS_paper_id] = ACL_paper_id
    outputFile = open('similarityUnique.txt', 'w')
    print len(pefectMatchMap)
    count = 1
    for key,value in hashmap.items():
        if value in pefectMatchMap:
            if (key == pefectMatchMap[value]):
                outputFile.write(value+'\t'+key+'\n')
            else:
                count = count + 1
                print 'duplicate'
        else:
            outputFile.write(value+'\t'+key+'\n')
    print count
    
def perfectFinalMatch():
    fileName = 'similarityUnique.txt'
    outputFile = open('perfectFinalMatch.txt', 'w')
    outputFile2 = open('perfectFinalMisMatch.txt', 'w')
    count = 1
    with open(fileName) as f:
        for line in f:
            print count
            elements = line.strip().split('\t')
            SSpaper_id = elements[0]
            ACLpaper_id = elements[1]
            SSfile = 'SS.txt'
            ACLFile = 'ACL.txt'
            title_acl = ''
            title_ss = ''
            with open(ACLFile) as f1:
                for line in f1:
                    aclid = line.strip().split('\t')[0]
                    if (aclid == ACLpaper_id):
                        title_acl = line.strip().split('\t')[2]
#                         outputFile.write(line.strip().split('\t')[2]+'\t')
                        break
            with open(SSfile) as f2:
                for line in f2:
                    data = json.loads(line)
                    ssid = str(data['id'])
                    if (ssid == SSpaper_id):
                        title_ss = str(data['title'])
#                         outputFile.write(data['title']+'\n')
                        break
            if (title_acl and title_ss):
                ratio = similar(title_acl, title_ss)
                outputFile.write(SSpaper_id + '\t' + ACLpaper_id + '\t' + title_ss + '\t' + title_acl + '\t' + str(ratio) + '\n')
            else:
                outputFile2.write(SSpaper_id+'\t'+ACLpaper_id+'\n')
            count = count + 1
    outputFile.close()   
    outputFile2.close()        
    
def main():
    perfectFinalMatch()
    
if __name__ == '__main__':
    main()
