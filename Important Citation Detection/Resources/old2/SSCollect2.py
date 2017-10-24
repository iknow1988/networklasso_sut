import json

def collectData():
    SS = set()
    have = set()
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[0].strip()
            SS.add(paper_id)

    SSFileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers-2017-02-21.json"
    outputFile = open('SS.txt', 'w')
    count = 1
    with open(SSFileName) as f:
        for line in f:
            data = json.loads(line)
            paper_id = data['id']
            if paper_id in SS:
                print count
                have.add(paper_id)
                outputFile.write(line)
                count = count + 1
    outputFile.close()
    for item in SS:
        if item not in have:
            print item
            
def collectData2():
    SS = set()
    have = set()
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[1].strip()
            SS.add(paper_id)
    print len(SS)
    SSFileName = "SS.txt"
    with open(SSFileName) as f:
        for line in f:
            data = json.loads(line)
            paper_id = data['id']
            if paper_id in SS:
                have.add(paper_id)
    for item in SS:
        if item not in have:
            print item
            
def collectData3():
    SS = set()
    count = 1
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[0].strip()
            if paper_id in SS:
                print line.strip()
            else:
                SS.add(paper_id)
            count = count + 1
    print count, len(SS)
    
                       
def main():
    collectData3()
            
if __name__ == '__main__':
    main()