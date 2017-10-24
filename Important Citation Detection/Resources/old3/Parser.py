import json
import urllib2
import pickle
import operator
from difflib import SequenceMatcher
import csv

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def createACLList():
    ACLFileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\ACL 2009\\Community-Detection-master\\aan\\release\\2013\\acl-metadata.txt"
    outputPaperFile = open('papersACL.txt', 'w')
    with open(ACLFileName) as f:
        data = f.read()
        papers = data.split("\n\n")
        for paper in papers:
            lines = paper.split('\n')
            paper_id = ''
            title = ''
            for line in lines :
                if (line.startswith( 'id' )):
                    paper_id = line.split('=')[1].strip().replace('{', '').replace('}', '')
                elif (line.startswith( 'title' )):
                    title = line.split('=')[1].strip().replace('{', '').replace('}', '')
            if paper_id and title:
                outputPaperFile.write(paper_id+'\t'+title)
                outputPaperFile.write('\n')
    outputPaperFile.close()
    
def sortACL():
    ACL = pickle.load(open("ACL.p", "rb"))
    otputFile = open('ACL.txt', 'w')
    sorted_x = sorted(ACL.items(), key=operator.itemgetter(1), reverse=False)
    for item in sorted_x:
        str1 = str(item[0])
        str2 = str(item[1]).strip().replace('"', '').replace('\'', '')
        if str1 and str2:
            otputFile.write(str1 + "\t" + str2 + '\n')
    otputFile.close()
       
def readACLList():
    ACLdataset = dict()
    fileName = "ACL.txt"
    with open(fileName) as f:
        for line in f:
            data = line.split('\t')
            paper_id = data[0]
            title = data[1].strip().replace('"', '').replace('\'', '')
            ACLdataset[paper_id] = title
    pickle.dump(ACLdataset, open("ACL.p", "wb"))

def createSemanticScholar():
    fileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers-2017-02-21.json"
    outputPaperFile = open('papersSemanticScholar.txt', 'w')
    count = 0
    with open(fileName) as f:
        for line in f:
            print count
            data = json.loads(line)
            paper_id = data['id']
            title = data['title']
            try:
                year = int(data['year'])
                if(2010<=year<=2013):
                    if paper_id and title:
                        try:
                            outputPaperFile.write(paper_id+'\t'+title)
                            outputPaperFile.write('\n')
                        except ValueError:
                            print 'err'
            except KeyError:
                print 'err2'
            count = count + 1
    outputPaperFile.close()

def readSemanticScholarPickle():
    SSdataset = dict()
    fileName = "sortedSSlatest.txt"
    count = 0 
    with open(fileName) as f:
        for line in f:
            data = line.split('\t')
            paper_id = data[0]
            title = data[1].strip().replace('"', '').replace('\'', '')
            SSdataset[paper_id] = title
            count = count + 1
    pickle.dump(SSdataset, open("SSlatest.p", "wb"))
    
def sortSS():
    SS = pickle.load(open("SSlatest.p", "rb"))
    print 'loaded'
    otputFile = open('sortedSSlatest.txt', 'w')
    sorted_x = sorted(SS.items(), key=operator.itemgetter(1), reverse=False)
    print 'writing start'
    for item in sorted_x:
        str1 = str(item[0])
        str2 = item[1].encode('ascii','ignore').strip().replace('"', '').replace('\'', '')
        if str1 and str2:
            otputFile.write(str1 + "\t" + str2 + '\n')
    otputFile.close()

def checkSemanticScholar():
    fileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers-2017-02-21.json"
    count = 0
    with open(fileName) as f:
        for line in f:
            print count
            data = json.loads(line)
            paper_id = data['id']
            if (paper_id == "0000c8f31b6058b1b42d9580c7afaa2e673a98c2"):
                title = data['title']
                print title
                print paper_id
                try:
                    year = int(data['year'])
                    print year
                except KeyError:
                    print 'err2'
                break
            count = count + 1
    

       
def findSimilairty():
    ACLfileName = "ACL.txt"
    fileName = "sortedSSlatest.txt"
    outputPaperFile = open('similarity.txt', 'w')
    count = 1
    found = 1
    with open(ACLfileName) as f1:
        for line1 in f1:
            print count
            data1 = line1.split('\t')
            if (len(data1) == 2):
                paper_id_acl = data1[0]
                title_acl = data1[1].strip()
                if paper_id_acl and title_acl:
                    with open(fileName) as f:
                        for line in f:
                            data = line.split('\t')
                            if (len(data) == 2):
                                paper_id = data[0]
                                title = data[1].strip()
                                if paper_id and title:
                                    test1 = title.split(' ')
                                    test2 = title_acl.split(' ')
                                    if(len(test1) == len(test2)):
                                        if(test1[0] == test2[0]):
                                            ratio = similar(title_acl, title)
                                            if(ratio > 0.9):
                                                print 'found : ', found
                                                outputPaperFile.write(paper_id + '\t' + paper_id_acl + '\t' + title + '\t' + title_acl + '\t'+ str(ratio))
                                                outputPaperFile.write('\n')
                                                found = found + 1
                                                break
                                    if(title[0] > title_acl[0]):
                                        break
            count = count + 1
                            
    outputPaperFile.close()
    
def createFile():
    fileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers-2017-02-21-sample.json"
    paperdict = set()
    authordict = set()
    outputPaperFile = open('papers.txt', 'w')
    outputAuthorFile = open('authors.txt', 'w')
    counter = 1;
    with open(fileName) as f:
        for line in f:
            print 'Line :', counter
            data = json.loads(line)
            paperid = data['id']
            if paperid not in paperdict:
                paperdict.add(paperid) 
                outputPaperFile.write(paperid)
                outputPaperFile.write('\n')
            authors = data['authors']
            for author in authors:
#                 print len(author['ids'])
                if len(author['ids']) != 0:
                    authorid = author['ids'][0]
                    if authorid not in authordict:
                        authordict.add(authorid) 
                        outputAuthorFile.write(authorid)
                        outputAuthorFile.write('\n')
            counter = counter + 1
    outputPaperFile.close()
    outputAuthorFile.close()

def crawlPapers():
    fileName = 'papers.txt'
    with open(fileName) as f:
        counter = 1
        for line in f:
            print 'Paper :', counter
            if line: 
                try:
                    line = line.strip()
                    link = 'http://api.semanticscholar.org/v1/paper/'+line
                    response = urllib2.urlopen(link)
                    data = response.read()
                    filename = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\Dataset\\Papers\\"+str(line) + ".json"
                    file_ = open(filename, 'w')
                    file_.write(data)
                    file_.close()
                except urllib2.HTTPError as err:
                    print counter,'\t' ,line,'\tNot found'   
                counter = counter + 1

def crawlAuthors():
    fileName = 'authors.txt'
    with open(fileName) as f:
        counter = 1
        for line in f:
            print 'Author :', counter
            if line: 
                try:
                    line = line.strip()
                    link = 'http://api.semanticscholar.org/v1/author/'+line
                    response = urllib2.urlopen(link)
                    data = response.read()
                    filename = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\Dataset\\Authors\\"+str(line) + ".json"
                    file_ = open(filename, 'w')
                    file_.write(data)
                    file_.close()
                except urllib2.HTTPError as err:
                    print counter,'\t' ,line,'\tNot found'   
                counter = counter + 1

def checkSimilarity():
    fileName = 'similarity3.txt'
    outputPaperFile = open('similarity.txt', 'w')
    with open(fileName) as f:
        counter = 1
        for line in f:
            elements = line.split("\t")
            SSid = elements[0]
            ACLid = elements[1]
            SStitle = elements[2].strip().lower()
            ACStitle = elements[3].strip().lower()
            outputPaperFile.write(SSid+'\t'+ACLid)
            outputPaperFile.write('\n')
            if (SStitle != ACStitle):
                print SStitle
                print ACStitle
                counter = counter + 1
    print counter
    outputPaperFile.close()
    
def findAnnotatedPapers():
    ACLdataset = set()
    with open('annotated.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvreader)
        for row in csvreader:
            print row
            id1 = row[1]
            id2 = row[2]
            ACLdataset.add(id1)
            ACLdataset.add(id2)
    pickle.dump(ACLdataset, open("Annotated.p", "wb"))      
    
def matchAnnotated():
    ACL = pickle.load(open("Annotated.p", "rb"))
    need = set()
    have = set()
    print "All papers: ",len(ACL)
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            id = line.split('\t')[1].strip()
            if id in ACL:
                have.add(id)
    for item in ACL:
        if item not in have:
            print item
            need.add(item)
    print len(need)
#     pickle.dump(need, open("need.p", "wb"))  
    
def matchWithACL():
    ACL = pickle.load(open("need.p", "rb"))
    print len(ACL)
    ACLFileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\ACL 2009\\Community-Detection-master\\aan\\release\\2013\\acl-metadata.txt"
    outputPaperFile = open('papersACLneeded.txt', 'w')
    with open(ACLFileName) as f:
        data = f.read()
        papers = data.split("\n\n")
        for paper in papers:
            lines = paper.split('\n')
            paper_id = ''
            title = ''
            for line in lines :
                if (line.startswith( 'id' )):
                    paper_id = line.split('=')[1].strip().replace('{', '').replace('}', '')
                elif (line.startswith( 'title' )):
                    title = line.split('=')[1].strip().replace('{', '').replace('}', '')
            if paper_id and title:
                if(paper_id in ACL):
                    outputPaperFile.write(paper_id+'\t'+title)
                    outputPaperFile.write('\n')
    outputPaperFile.close()

def checkyears():
    ACL = pickle.load(open("need.p", "rb"))
    yearDict = dict()
    print len(ACL)
    ACLFileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\ACL 2009\\Community-Detection-master\\aan\\release\\2013\\acl-metadata.txt"
    count = 0
    with open(ACLFileName) as f:
        data = f.read()
        papers = data.split("\n\n")
        for paper in papers:
            lines = paper.split('\n')
            paper_id = ''
            title = ''
            year = ''
            for line in lines :
                if (line.startswith( 'id' )):
                    paper_id = line.split('=')[1].strip().replace('{', '').replace('}', '')
                    continue
                elif (line.startswith( 'title' )):
                    title = line.split('=')[1].strip().replace('{', '').replace('}', '')
                    continue
                elif (line.startswith( 'year' )):
                    year = line.split('=')[1].strip().replace('{', '').replace('}', '')
                    continue
            if paper_id and title and year:
                if(paper_id in ACL):
                    count = count + 1
                    if year in yearDict:
                        yearDict[year] = yearDict[year] + 1
                    else:
                        yearDict[year] = 1
    print count                   
       
def readneededACLList():
    ACLdataset = dict()
    fileName = "papersACLneeded.txt"
    with open(fileName) as f:
        for line in f:
            data = line.split('\t')
            paper_id = data[0].strip()
            title = data[1].strip()
            ACLdataset[paper_id] = title
    pickle.dump(ACLdataset, open("ACLneeded.p", "wb"))

def neededSemanticScholar():
    fileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers-2017-02-21.json"
    outputPaperFile = open('papersSemanticScholar.txt', 'w')
    count = 0
    with open(fileName) as f:
        for line in f:
            print count
            data = json.loads(line)
            paper_id = data['id']
            title = data['title']
            try:
                if paper_id and title:
                    try:
                        outputPaperFile.write(paper_id+'\t'+title)
                        outputPaperFile.write('\n')
                    except ValueError:
                        print 'err'
            except KeyError:
                print 'err2'
            count = count + 1
    outputPaperFile.close()
    
def checkCommon():
    ids = set()
    fileName = 'id_similarity.txt'
    outputPaperFile = open('find_papers.txt', 'w')
    with open(fileName) as f:
        counter = 1
        for line in f:
            elements = line.split("\t")
            SSid = elements[1].strip()
            ids.add(SSid)
    
    fileName = 'ACL.txt'
    with open(fileName) as f:
        counter = 1
        for line in f:
            elements = line.split("\t")
            id = elements[0]
            title = elements[1].strip()
            if id not in ids:
                outputPaperFile.write(id+'\t'+title)
                outputPaperFile.write('\n')
    print counter
    outputPaperFile.close()
        
def main():
    findSimilairty()

if __name__ == '__main__':
    main()