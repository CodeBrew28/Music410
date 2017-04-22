import numpy as np
from scipy.special import psi
import json, string, math, random, re, codecs, jieba

# Shamelessly copied from  https://github.com/laserwave/LDA-Variational-EM/blob/master/main.py

class Document:
    def __init__(self, itemIdList, itemCountList, wordCount):
        self.itemIdList = itemIdList
        self.itemCountList = itemCountList
        self.wordCount = wordCount

def preprocessing(data):
        # print(data)
    docs = []
    word2id = {}
    id2word = {}
    currentWordId = 0
    for document in data:
        document = document['lyrics']
        word2Count = {}
        segList = jieba.cut(document)
        for word in segList: 
            word = word.lower().strip()
            cleaned_word = ""
            for letter in word:
                if letter not in string.punctuation:
                    cleaned_word += letter
            word = cleaned_word
            if len(word) > 1 and not re.search('[0-9],', word) and word not in stopwords:
                if word not in word2id:
                    word2id[word] = currentWordId
                    id2word[currentWordId] = word
                    currentWordId += 1
                if word in word2Count:
                    word2Count[word] += 1
                else:
                    word2Count[word] = 1
        itemIdList = []
        itemCountList = []
        wordCount = 0
        for word in word2Count.keys():
            itemIdList.append(word2id[word])
            itemCountList.append(word2Count[word])
            wordCount += word2Count[word]
        docs.append(Document(itemIdList, itemCountList, wordCount))
    return docs, word2id, id2word


def maxItemNum():
    num = 0
    for d in range(0, N):
        if len(docs[d].itemIdList) > num:
            num = len(docs[d].itemIdList)
    return num

def initialLdaModel():
    for z in range(0, K):
        for w in range(0, M):
            nzw[z, w] += 1.0/M + random.random()
            nz[z] += nzw[z, w]
    updateVarphi()    

# update model parameters : varphi (the update of alpha is ommited)
def updateVarphi():
    for z in range(0, K):
        for w in range(0, M):
            if(nzw[z, w] > 0):
                varphi[z, w] = math.log(nzw[z, w]) - math.log(nz[z])
            else:
                varphi[z, w] = -100

# update variational parameters : gamma and phi
def variationalInference(docs, d, gamma, phi):
    phisum = 0
    oldphi = np.zeros([K])
    digamma_gamma = np.zeros([K])
    
    for z in range(0, K):
        gamma[d][z] = alpha + docs[d].wordCount * 1.0 / K
        digamma_gamma[z] = psi(gamma[d][z])
        for w in range(0, len(docs[d].itemIdList)):
            phi[w, z] = 1.0 / K

    for iteration in range(0, iterInference):
        for w in range(0, len(docs[d].itemIdList)):
            phisum = 0
            for z in range(0, K):
                oldphi[z] = phi[w, z]
                phi[w, z] = digamma_gamma[z] + varphi[z, docs[d].itemIdList[w]]
                if z > 0:
                    phisum = math.log(math.exp(phisum) + math.exp(phi[w, z]))
                else:
                    phisum = phi[w, z]
            for z in range(0, K):
                phi[w, z] = math.exp(phi[w, z] - phisum)
                gamma[d][z] =  gamma[d][z] + docs[d].itemCountList[w] * (phi[w, z] - oldphi[z])
                digamma_gamma[z] = psi(gamma[d][z])


def getTopicWords(docs, word2id, id2word):
    
    # initialization of the model parameter varphi, the update of alpha is ommited
    initialLdaModel()

    # variational EM Algorithm
    for iteration in range(0, iterEM): 
        nz = np.zeros([K])
        nzw = np.zeros([K, M])
        alphaSS = 0
        # E-Step
        for d in range(0, N):
            variationalInference(docs, d, gamma, phi)
            gammaSum = 0
            for z in range(0, K):
                gammaSum += gamma[d, z]
                alphaSS += psi(gamma[d, z])
            alphaSS -= K * psi(gammaSum)

            for w in range(0, len(docs[d].itemIdList)):
                for z in range(0, K):
                    nzw[z][docs[d].itemIdList[w]] += docs[d].itemCountList[w] * phi[w, z]
                    nz[z] += docs[d].itemCountList[w] * phi[w, z]

        # M-Step
        updateVarphi()

    # calculate up to 10 terms of each topic
    topicwords = {}
    maxTopicWordsNum = 10
    for z in range(0, K):
        ids = varphi[z, :].argsort()
        for j in ids:
            topicwords[id2word[j]] = 1 
    return topicwords


file = codecs.open('stopwords.dic','r','utf-8')
stopwords = [line.strip() for line in file]
file.close()
with open('data/2015.json') as data_file:  
    data_set = json.load(data_file)


songs = {}
for i in range(0, len(data_set)):
    data = [data_set[i]]
    docs, word2id, id2word = preprocessing(data) 
    N = len(docs) # number of documents for training
    M = len(word2id) # number of distinct terms
    K = 10 # number of topic
    iterInference = 20  # iteration times of variational inference, judgment of the convergence by calculating likelihood is ommited
    iterEM = 20 # iteration times of variational EM algorithm, judgment of the convergence by calculating likelihood is ommited
    alpha = 5 # initial value of hyperparameter alpha
    alphaSS = 0 # sufficient statistic of alpha
    varphi = np.zeros([K, M]) # the topic-word distribution (beta in D. Blei's paper)
    nzw = np.zeros([K, M]) # topic-word count, this is a sufficient statistic to calculate varphi
    nz = np.zeros([K]) # topic count, sum of nzw with w ranging from [0, M-1], for calculating varphi
    gamma = np.zeros([N, K])  # inference parameter gamma
    phi = np.zeros([maxItemNum(), K])  # inference parameter phi
    getTopicWords(docs, word2id, id2word)
    songs[data[0]['title'] ] = getTopicWords(docs, word2id, id2word).keys()

print(songs)


import spacy

color_assciation = {}

nlp = spacy.load('en')
w2 = nlp(unicode("ocean"))

w1 = nlp(unicode("red"))
similarity_rating = w1.similarity(w2)
color_assciation["red"] = similarity_rating

w1 = nlp(unicode("orange"))
similarity_rating = w1.similarity(w2)
color_assciation["orange"] = similarity_rating

w1 = nlp(unicode("yellow"))
similarity_rating = w1.similarity(w2)
color_assciation["yellow"] = similarity_rating

w1 = nlp(unicode("green"))
similarity_rating = w1.similarity(w2)
color_assciation["green"] = similarity_rating

w1 = nlp(unicode("blue"))
similarity_rating = w1.similarity(w2)
color_assciation["blue"] = similarity_rating

w1 = nlp(unicode("purple"))
similarity_rating = w1.similarity(w2)
color_assciation["purple"] = similarity_rating


print(color_assciation)