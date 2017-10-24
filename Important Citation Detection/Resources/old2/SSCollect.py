import urllib2

def main():
    SS = set()
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[0].strip()
            SS.add(paper_id)
    counter = 1
    for item in SS:
#         print counter
        try:
            link = 'http://api.semanticscholar.org/v1/paper/'+item
            response = urllib2.urlopen(link)
            data = response.read()
            filename = "C:\\Users\\kadnan\\Google Drive\\Research\\Projects\\11. SemanticScholarParser\\Dataset\\Semantic Scholar\Papers\\"+str(item) + ".json"
            file_ = open(filename, 'w')
            file_.write(data)
            file_.close()
        except urllib2.HTTPError as err:
            print counter,'\t' ,item,'\tNot found'
        counter = counter + 1
        
if __name__ == '__main__':
    main()