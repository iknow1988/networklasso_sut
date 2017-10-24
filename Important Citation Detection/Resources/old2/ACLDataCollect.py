
def main():
    ACL = set()
    have = set()
    fileName = 'similarity.txt'
    with open(fileName) as f:
        for line in f:
            paper_id = line.split('\t')[1].strip()
            ACL.add(paper_id)
    
    ACLFileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\ACL 2009\\Community-Detection-master\\aan\\release\\2013\\acl-metadata.txt"
    outputFile = open('ACL.txt', 'w')
    count = 1
    with open(ACLFileName) as f:
        data = f.read()
        papers = data.split("\n\n")
        for paper in papers:
            print count
            lines = paper.split('\n')
            paper_id = ''
            title = ''
            author = ''
            venue = ''
            year = ''
            for line in lines :
                if (line.startswith('id')):
                    paper_id = line.split('=')[1].strip().replace('{', '').replace('}', '')
                    if paper_id not in ACL:
                        break
                    continue
                elif (line.startswith('author')):
                    author = line.split('=')[1].strip().replace('{', '').replace('}', '')
                    continue
                elif (line.startswith('title')):
                    title = line.split('=')[1].strip().replace('{', '').replace('}', '')
                    continue
                elif (line.startswith('venue')):
                    venue = line.split('=')[1].strip().replace('{', '').replace('}', '')
                    continue
                elif (line.startswith('year')):
                    year = line.split('=')[1].strip().replace('{', '').replace('}', '')
                    continue
            if paper_id in ACL:
                have.add(paper_id)
                outputFile.write(paper_id + '\t' + author + '\t' + title + '\t' + venue + '\t' + year +'\n')
            count = count + 1
        for item in ACL:
            if item not in have:
                print item
            
    outputFile.close()       
    
if __name__ == '__main__':
    main()
