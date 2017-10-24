import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

def createDataset():
    dataset_file = 'dataset.csv'
    dataset = list()
    with open(dataset_file, 'rb') as f:
        f.readline()
        for line in f:
            data = {}
            elements = line.strip().split(',')
            data['citedPaper'] = elements[0].strip()
            data['citingPaper'] = elements[1].strip()
            data['authorSimilarity'] = elements[2].strip()
            data['referenceSimilarity'] = elements[3].strip()
            data['titleSimilarity'] = elements[4].strip()
            data['abstractSimilarity'] = elements[5].strip()
            data['keyPhraseSimilarity'] = elements[6].strip()
            data['label'] = elements[7].strip()
            dataset.append(data)
    pickle.dump(dataset, open("dataset.p", "wb"))

def getDatasets():
    dataset = pickle.load(open("dataset_small.p", "rb"))
    samples = list()
    for data in dataset:
        sample = np.ones(6) * -1
        sample[0] = data['authorSimilarity']
        sample[1] = data['referenceSimilarity']
        sample[2] = data['titleSimilarity']
        sample[3] = data['abstractSimilarity']
        sample[4] = data['keyPhraseSimilarity']
        sample[5] = data['label']
        samples.append(sample)
    samples = np.asarray(samples)
    imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
    imp.fit(samples)
#     features = np.concatenate((samples[:,0:1],samples[:,3:4],samples[:,4:5]),axis=1)
    features = samples[:,4:5]
    labels = samples[:, 5:6]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)
    return X_train, X_test, y_train.ravel(), y_test.ravel(), features, labels.ravel()
    
def main():
    X_train, X_test, y_train, y_test, X_all, y_all = getDatasets()
    print 'Train:', len(X_train), 'Test:', len(X_test)
    print "Number of important citation in Training and Testing respectively: ", int(sum(y_train)), int(sum(y_test))
    
#     print '*****SVM****'
#     clf = svm.SVC(C=0.75, kernel='rbf', gamma='auto', probability=True, class_weight={1: 6})
# #     clf = svm.SVC(C=0.75, kernel='rbf', gamma='auto', probability=True, class_weight={1: 13})
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print "Predicted Important citations: ", int(sum(y_pred))
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     print "TP,TN,FP,FN", tp, tn, fp, fn
# #     print 'Accuracy:', accuracy_score(y_test, y_pred, normalize=True), 'Recall:', recall_score(y_test, y_pred, average='micro'), 'F1:', f1_score(y_test, y_pred, average='micro')
#     precision = 1.0 * tp / (tp + fp)
#     recall = 1.0 * tp / (tp + fn)
#     accuracy = 1.0 * (tp + tn) / len(X_test)
#     f1 = 2.0 * (precision * recall) / (precision + recall)
#     print 'Accuracy:', accuracy, 'Recall:', recall , 'Precision:', precision, 'F1:', f1 
#     y_pred_svm = clf.predict_proba(X_test)[:, 1]
#     fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
#     precision_svm, recall_svm, _ = precision_recall_curve(y_test, y_pred_svm)
#     scores = cross_val_score(clf, X_all, y_all, cv=3)
#     print scores
    
    print '****Random Forest****'
    clf = RandomForestClassifier(n_jobs=-1, random_state=0, class_weight={1: 6})
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print "Predicted Important citations: ", int(sum(y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print "TP,TN,FP,FN", tp, tn, fp, fn
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    accuracy = 1.0 * (tp + tn) / len(X_test)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    print 'Accuracy:', accuracy, 'Recall:', recall , 'Precision:', precision, 'F1:', f1 
    y_pred_rf = clf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)
#     scores = cross_val_score(clf, X_all, y_all, cv=3)
#     print scores
    
    print '****Logistic Regression****'
#     clf = linear_model.LogisticRegression(C=1e5, class_weight={1: 13})
    clf = linear_model.LogisticRegression(C=0.75, class_weight={1: 6})
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print "Predicted Important citations: ", int(sum(y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print "TP,TN,FP,FN", tp, tn, fp, fn
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    accuracy = 1.0 * (tp + tn) / len(X_test)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    print 'Accuracy:', accuracy, 'Recall:', recall , 'Precision:', precision, 'F1:', f1 
    y_pred_lr = clf.decision_function(X_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_lr)
#     scores = cross_val_score(clf, X_all, y_all, cv=3)
#     print scores
    
    print '****naive bayes****'
    clf = GaussianNB()
    clf.fit(X_train, y_train)
#     print clf.class_prior_
    y_pred = clf.predict(X_test)
    print "Predicted Important citations: ", int(sum(y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print "TP,TN,FP,FN", tp, tn, fp, fn
    precision = 1.0 * tp / (tp + fp)
    recall = 1.0 * tp / (tp + fn)
    accuracy = 1.0 * (tp + tn) / len(X_test)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    print 'Accuracy:', accuracy, 'Recall:', recall , 'Precision:', precision, 'F1:', f1 
    y_pred_nb = clf.predict_proba(X_test)[:, 1]
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_nb)
    precision_nb, recall_nb, _ = precision_recall_curve(y_test, y_pred_nb)
#     scores = cross_val_score(clf, X_all, y_all, cv=3)
#     print scores
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.plot(fpr_rf, tpr_rf, label='Random Forest')
#     plt.plot(fpr_svm, tpr_svm, label='SVM')
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
    plt.plot(fpr_nb, tpr_nb, label='Naive Bayes')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
#     plt.show()
    
    plt.figure(2)
#     plt.plot(recall_svm, precision_svm, label='SVM')
    plt.plot(recall_rf, precision_rf, label='Random Forest')
    plt.plot(recall_lr, precision_lr, label='Logistic Regression')
    plt.plot(recall_nb, precision_nb, label='Naive Bayes')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')
#     plt.show()

if __name__ == '__main__':
    main()
