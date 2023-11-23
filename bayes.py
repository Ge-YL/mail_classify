import numpy as np
import re
import random


def textParse(input_string):
    list_of_tokens = re.split(r'\w+', input_string)
    return [tok.lower() for tok in list_of_tokens if len(list_of_tokens) > 2]


def creatVocablist(doclist):
    vocab_set = set([])
    for document in doclist:
        vocab_set = vocab_set | set(document) #并上之前的词
    return list(vocab_set)  #再转换成list


def setofword2vec(vocablist, inputset):
    returnvce = [0] * len(vocablist)
    for word in inputset:
        #如果email中的词出现在词典中，则对应位置设为1
        if word in vocablist:
            returnvce[vocablist.index(word)] = 1
    return returnvce


def trainNB(trainMat, trainClass):
    numTrainDocs = len(trainMat) #获取训练集email的数量
    numWords = len(trainMat[0]) #特征向量的长度，其实就是词典的大小
    p1 = sum(trainClass) / float(numTrainDocs)  # 垃圾邮件的概率，正常邮件就为1-p1
    p0Num = np.ones((numWords)) #laplace smoothing，防止某个单词第一次出现，使得概率为0
    p1Num = np.ones((numWords))
    p0Denum = 2 #它是做分母的，也一样操作，一般分为多少类就为多少，|V|
    p1Denum = 2
    for i in range(numTrainDocs):
        if trainClass[i] == 1:  # 表示垃圾邮件
            p1Num += trainMat[i]  # 垃圾邮件中相应单词在所有垃圾邮件中出现的次数，这边一封邮件中出现多次一个单词呢？？？
            p1Denum += sum(trainMat[i])  # 垃圾邮件中的总词数
        else:
            p0Num += trainMat[i]  # 正常邮件中相应单词在所有正常邮件中出现的次数
            p0Denum += sum(trainMat[i]) # 正常邮件中的总词数
    p1Vec = np.log(p1Num / p1Denum)
    p0Vec = np.log(p0Num / p0Denum)
    return p0Vec, p1Vec, p1


def classifyNB(wordVec, p0Vec, p1Vec, p1_class):
    p1 = np.log(p1_class) + sum(p1Vec * wordVec)
    p0 = np.log(1.0 - p1_class) + sum(p0Vec * wordVec)
    if p0 > p1:
        return 0
    else:
        return 1


def spam():
    doclist = []
    classlist = []
    #读入数据和获取标签
    for i in range(1, 26):
        wordlist = textParse(open('email/spam/%d.txt' % i, 'r').read())
        doclist.append(wordlist)
        classlist.append(1)

        wordlist = textParse(open('email/ham/%d.txt' % i, 'r').read())
        doclist.append(wordlist)
        classlist.append(0)

    # 获取词典
    vocablist = creatVocablist(doclist)
    # 分成训练集和测试集，训练集用于求先验概率
    trainSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    #测试集特征矩阵及其标签
    trainMat = []
    trainClass = []

    for docIndex in trainSet:
        # 测试集特征矩阵及其对应标签
        trainMat.append(setofword2vec(vocablist, doclist[docIndex]))
        trainClass.append(classlist[docIndex])
    #训练集先验概率计算
    p0Vec, p1Vec, p1 = trainNB(np.array(trainMat), np.array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVec = setofword2vec(vocablist, doclist[docIndex])
        if classifyNB(np.array(wordVec), p0Vec, p1Vec, p1) != classlist[docIndex]:
            errorCount += 1

    print('当前10个测试样本，错了：', errorCount, '个')


if __name__ == '__main__':
    spam()
