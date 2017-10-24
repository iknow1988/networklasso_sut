import json

def collectData():
    SS = set()
    have = set()
    fileName = 'additional.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[0].strip()
            SS.add(paper_id)

    SSFileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers-2017-02-21.json"
    outputFile = open('SS_additional.txt', 'w')
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
    fileName = 'additional.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.strip()
            SS.add(paper_id)
    print len(SS)
    SSFileName = "SS_additional.txt"
    with open(SSFileName) as f:
        for line in f:
            data = json.loads(line)
            paper_id = data['id']
            if paper_id in SS:
                have.add(paper_id)
    outputFile = open('SS_additional_failed.txt', 'w')
    for item in SS:
        if item not in have:
            outputFile.write(item+'\n')
    
    outputFile.close()
                      
def main():
    collectData2()
            
if __name__ == '__main__':
    main()