import json
import urllib2
import os.path

def createFile():
    fileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers-2017-02-21.json"
    paperdict = set()
    authordict = set()
    outputPaperFile = open("C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers.txt", 'w')
    outputAuthorFile = open("C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\authors.txt", 'w')
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
    fileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers.txt"
    with open(fileName) as f:
        counter = 1
        for line in f:
            print 'Paper :', counter
            if line: 
                try:
                    line = line.strip()
                    filename = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\Dataset\\Papers\\"+str(line) + ".json"
                    if not os.path.exists(filename):
                        link = 'http://api.semanticscholar.org/v1/paper/'+line
                        response = urllib2.urlopen(link)
                        data = response.read() 
                        file_ = open(filename, 'w')
                        file_.write(data)
                        file_.close()
                except urllib2.HTTPError as err:
                    print counter,'\t' ,line,'\tNot found'   
                counter = counter + 1

def crawlAuthors():
    fileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\authors.txt"
    with open(fileName) as f:
        counter = 1
        for line in f:
            print 'Author :', counter
            if line: 
                try:
                    line = line.strip()
                    filename = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\Dataset\\Authors\\"+str(line) + ".json"
                    link = 'http://api.semanticscholar.org/v1/author/'+line
                    response = urllib2.urlopen(link)
                    data = response.read()
                    file_ = open(filename, 'w')
                    file_.write(data)
                    file_.close()
                except urllib2.HTTPError as err:
                    print counter,'\t' ,line,'\tNot found'   
                counter = counter + 1
                               
def main():
#     createFile()
    crawlPapers()
#     crawlAuthors()
    
if __name__ == '__main__':
    main()