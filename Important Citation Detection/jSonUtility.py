import os
import json
from scipy.odr.odrpack import Output

def writeJsonFile():
    directory = "Papers\\"
    files = os.listdir(directory)
    inDictionary = set()
    for f in files:
        inDictionary.add(str(f.split('.')[0]).strip())
        
    SSFileName = "C:\\Users\\kadnan\\Desktop\\Semantic Scholar\\papers-2017-02-21.json"
    outputFile = open('SS.txt', 'w')
    outputFile2 = open('failed.txt', 'w')
    paper = 0
    found = 0
    papers = set()
    with open(SSFileName) as f:
        for line in f:
            data = json.loads(line)
            paper_id = data['id']
            if paper_id in inDictionary:
                outputFile.write(line.strip()+'\n')
                found = found + 1
                papers.add(paper_id)
            paper = paper + 1
            print 'paper:',paper,', found:',found
    
    for paper in inDictionary:
        if paper not in papers:
            outputFile2.write(paper+'\n')
            
    outputFile.close()
    outputFile2.close()

def createJSON():
    inDictionary = set()
    with open('failed.txt') as f:
        for line in f:
            paperId = line.strip()
            inDictionary.add(paperId)
    
    directory = "Papers\\"
    files = os.listdir(directory)
    jsonObjects = list()
    for f in files:
        fileName = f
        paperId = str(fileName.split('.')[0]).strip()
        if paperId in inDictionary:
            if (os.path.isfile(os.path.join(directory, paperId + '.json'))):
                x_file = open(os.path.join(directory, paperId + '.json'), "r")
                data = json.load(x_file)
                jsonData = {}
                for attribute, value in data.iteritems():
                    if (attribute == 'authors'):
                        authorObjects = list()
                        for value11 in value:
                            authorData = {}
                            for attribute1, value1 in value11.iteritems():
                                if (attribute1 == 'authorId'):
                                    authorData['ids'] = value1
                                elif (attribute1 == 'name'):
                                    authorData['name'] = value1
    #                         author_data = json.dumps(authorData, sort_keys=True)
                            authorObjects.append(authorData)  
                        jsonData['authors'] =  authorObjects
                    elif(attribute == 'paperId'):
                        jsonData['id'] =  value
                    elif(attribute == 'references'):
                        jsonData['outCitations'] =  value
                    elif(attribute == 'title'):
                        jsonData['title'] =  value 
                    elif(attribute == 'venue'):
                        jsonData['venue'] =  value
                    elif(attribute == 'year'):
                        jsonData['year'] = value
                    elif(attribute == 'citationVelocity'):
                        jsonData['citationVelocity'] =  value
                    elif(attribute == 'influentialCitationCount'):
                        jsonData['influentialCitationCount'] =  value
                jsonData['keyPhrases'] = ''
                jsonData['paperAbstract'] = ''
                jsonData['pdfUrls'] = ''
                jsonData['s2Url'] = ''
                jsonData['inCitations'] = ''
                json_data = json.dumps(jsonData, sort_keys=True)
                jsonObjects.append(json_data)
            else:
                print paperId, 'not found'
    
    outfile = open('SSnew.txt', 'w')
    for jsonObject in jsonObjects:
#             json.dump(jsonObject.replace('\\"',"\""), outfile)
        outfile.write(jsonObject)
        outfile.write('\n')
    outfile.close()

def modifyJson():
    directory = "Papers\\"
    jsonObjects = list()
    with open('SS.txt') as f:
        for line in f:
            data = json.loads(line)
            paperId = data['id']
            x_file = open(os.path.join(directory, paperId + '.json'), "r")
            newData = json.load(x_file)
            citationVelocity = newData['citationVelocity']
            influentialCitationCount = newData['influentialCitationCount']
            data['citationVelocity'] = citationVelocity
            data['influentialCitationCount'] = influentialCitationCount
            json_data = json.dumps(data, sort_keys=True)
            jsonObjects.append(json_data)
    outfile = open('SSmodified.txt', 'w')
    for jsonObject in jsonObjects:
        outfile.write(jsonObject)
        outfile.write('\n')
    outfile.close()

def readNewJson():
    count = 1
    with open('SS.txt') as f:
        for line in f:
            data = json.loads(line)
            paperId = data['id']
            print count,paperId
            count = count + 1
def main():
    createJSON()
    
if __name__ == '__main__':
    main()