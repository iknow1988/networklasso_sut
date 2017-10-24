import json
import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, string, numpy
stemmer = nltk.stem.porter.PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer

def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def StemNormalize(text):
    return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def commonAuthors():
    outputFile = open('commonAuthors.csv', 'w')
    count = 1
    with open('citations_dataset.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            citedPaper = elements[0].strip()
            citingPaper = elements[1].strip()
            paper1 = False
            paper2 = False
            citedPaperAuthors = list()
            citingPaperAuthors = list()
            with open('SS.txt') as f1:
                for line1 in f1:
                    data = json.loads(line1)
                    paperId = data['id']
                    if (not paper1 or not paper2):
                        if (paperId == citedPaper):
                            authors = data['authors']
                            for author in authors:
                                temp = author['ids']
                                if(type(temp) is list):
                                    if(len(temp))>0:
                                        citedPaperAuthors.append(str(temp[0]))
                                else:
#                                     print 'here1', temp
                                    citedPaperAuthors.append(temp)
                            paper1 = True
                        elif(paperId == citingPaper):
                            authors = data['authors']
                            for author in authors:
                                temp = author['ids']
                                if(type(temp) is list):
                                    if(len(temp))>0:
                                        citingPaperAuthors.append(str(temp[0]))
                                else:
#                                     print 'here2', temp
                                    citingPaperAuthors.append(temp)
                            paper2 = True
                    else:
                        break
            authorList =  set (citedPaperAuthors).union(set(citingPaperAuthors))
            commonAuthors = set (citedPaperAuthors) & set(citingPaperAuthors)
            similarity = 1.0 * (0 if len(commonAuthors) == 0 else len(commonAuthors))/(1 if len(authorList) == 0 else len(authorList))
            print count, citedPaper, citingPaper, ':', similarity, len(commonAuthors), len(authorList),citedPaperAuthors, citingPaperAuthors
            outputFile.write(citedPaper+','+citingPaper+','+str(similarity)+'\n')
            count = count  + 1
    outputFile.close()

def commonReferences():
    outputFile = open('commonReferences.csv', 'w')
    count = 1
    with open('citation_dataset_small.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            citedPaper = elements[0].strip()
            citingPaper = elements[1].strip()
            paper1 = False
            paper2 = False
            citedPaperReferences = list()
            citingPaperReferences = list()
            with open('SS.txt') as f1:
                for line1 in f1:
                    data = json.loads(line1)
                    paperId = data['id']
                    if (not paper1 or not paper2):
                        if (paperId == citedPaper):
                            references = data['outCitations']
                            for reference in references:
                                if(type(reference) is unicode):
                                    citedPaperReferences.append(reference)
                                elif(type(reference) is dict):
                                    for attribute, value in reference.iteritems():
                                            if (attribute == 'paperId'):
                                                citedPaperReferences.append(value)
                            paper1 = True
                        elif(paperId == citingPaper):
                            references = data['outCitations']
                            for reference in references:
                                if(type(reference) is unicode):
                                    citingPaperReferences.append(reference)
                                elif(type(reference) is dict):
                                    for attribute, value in reference.iteritems():
                                            if (attribute == 'paperId'):
                                                citingPaperReferences.append(value)
                            paper2 = True
                    else:
                        break
#             print citedPaperReferences, citingPaperReferences
            referenceList =  set (citedPaperReferences).union(set(citingPaperReferences))
            commonReferences = set (citedPaperReferences) & set(citingPaperReferences)
            similarity = 1.0 * (0 if len(commonReferences) == 0 else len(commonReferences))/(1 if len(referenceList) == 0 else len(referenceList))
            print count,len(citedPaperReferences), len(citingPaperReferences), ':', similarity, len(commonReferences), len(referenceList)
            outputFile.write(citedPaper+','+citingPaper+','+str(similarity)+'\n')
            count = count  + 1
    outputFile.close()

def titleSimilarity():
    outputFile = open('titleSimilarity.csv', 'w')
    count = 1
    with open('citation_dataset_small.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            citedPaper = elements[0].strip()
            citingPaper = elements[1].strip()
            paper1 = False
            paper2 = False
            citedPaperTitle = ''
            citingPaperTitle= ''
            with open('SS.txt') as f1:
                for line1 in f1:
                    data = json.loads(line1)
                    paperId = data['id']
                    if (not paper1 or not paper2):
                        if (paperId == citedPaper):
                            citedPaperTitle = data['title']
                            paper1 = True
                        elif(paperId == citingPaper):
                            citingPaperTitle = data['title']
                            paper2 = True
                    else:
                        break
#             print citedPaperReferences, citingPaperReferences
            TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
            textlist = list()
            textlist.append(citedPaperTitle)
            textlist.append(citingPaperTitle)
            tfidf = TfidfVec.fit_transform(textlist)
            similarity = (tfidf * tfidf.T).toarray()[0,1]
            similarity2 = cosine.get_result(citedPaperTitle, citingPaperTitle)
            print count,citedPaperTitle, citingPaperTitle, ':', similarity, similarity2
            outputFile.write(citedPaper+','+citingPaper+','+str(similarity)+','+str(similarity2)+'\n')
            count = count  + 1
    outputFile.close()
    
def abstractSimilarity():
    outputFile = open('abstractSimilarity.csv', 'w')
    count = 1
    with open('citations_dataset.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            citedPaper = elements[0].strip()
            citingPaper = elements[1].strip()
            paper1 = False
            paper2 = False
            citedPaperAbstract = ''
            citingPaperAbstract= ''
            with open('SS.txt') as f1:
                for line1 in f1:
                    data = json.loads(line1)
                    paperId = data['id']
                    if (not paper1 or not paper2):
                        if (paperId == citedPaper):
                            citedPaperAbstract = data['paperAbstract']
                            paper1 = True
                        elif(paperId == citingPaper):
                            citingPaperAbstract = data['paperAbstract']
                            paper2 = True
                    else:
                        break
#             print citedPaperReferences, citingPaperReferences
            if(citedPaperAbstract and citingPaperAbstract):
                TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
                textlist = list()
                textlist.append(citedPaperAbstract)
                textlist.append(citingPaperAbstract)
                tfidf = TfidfVec.fit_transform(textlist)
                similarity = (tfidf * tfidf.T).toarray()[0,1]
                print count, ':', similarity
                outputFile.write(citedPaper+','+citingPaper+','+str(similarity)+'\n')
            else:
                outputFile.write(citedPaper+','+citingPaper+','+str(-1)+'\n')
                print count, ': Not found abstract'
#             print count
            count = count  + 1
    outputFile.close()
    
def keyphraseSimilarity():
    outputFile = open('keyphraseSimilarity.csv', 'w')
    count = 1
    with open('citation_dataset_small.csv', 'rb') as f:
        f.readline()
        for line in f:
            elements = line.strip().split(',')
            citedPaper = elements[0].strip()
            citingPaper = elements[1].strip()
            paper1 = False
            paper2 = False
            citedPaperkeyPhrases = list()
            citingPaperkeyPhrases= list()
            with open('SS.txt') as f1:
                for line1 in f1:
                    data = json.loads(line1)
                    paperId = data['id']
                    if (not paper1 or not paper2):
                        if (paperId == citedPaper):
                            temp = data['keyPhrases']
                            if(len(temp)>0):
                                for keys in temp:
                                    citedPaperkeyPhrases.append(keys)
                            paper1 = True
                        elif(paperId == citingPaper):
                            temp = data['keyPhrases']
                            if(len(temp)>0):
                                for keys in temp:
                                    citingPaperkeyPhrases.append(keys)
                            paper2 = True
                    else:
                        break
#             print citedPaperReferences, citingPaperReferences
            if(len(citedPaperkeyPhrases)>0 and len(citingPaperkeyPhrases)>0):
                keyphraseList =  set (citedPaperkeyPhrases).union(set(citingPaperkeyPhrases))
                commonkeys = set (citedPaperkeyPhrases) & set(citingPaperkeyPhrases)
                similarity = 1.0 * (0 if len(commonkeys) == 0 else len(commonkeys))/(1 if len(keyphraseList) == 0 else len(keyphraseList))
                print count,citedPaperkeyPhrases, citingPaperkeyPhrases, ':', similarity, len(commonkeys), len(keyphraseList)
                outputFile.write(citedPaper+','+citingPaper+','+str(similarity)+'\n')
            else:
                outputFile.write(citedPaper+','+citingPaper+','+str(-1)+'\n')
                print count, ': Not found abstract'
#             print count
            count = count  + 1
    outputFile.close()   
def main():
    abstractSimilarity()
if __name__ =='__main__':
    main()