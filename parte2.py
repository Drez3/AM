import csv
import functools
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as proc
import random
import scipy
import operator


def main():
    # separando as duas views
    print('--- inciando ---')
    shape_view_fields = ['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD']

    rgb_view_fields = ['INTENSITY-MEAN', 'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN',
                       'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN', 'HUE-MEAN']

    shape_view_table = pd.read_csv('tabela_tratada_para_testes.txt', usecols=shape_view_fields)
    # print(shape_view_table)

    rgb_view_table = pd.read_csv('tabela_tratada_para_testes.txt', usecols=rgb_view_fields)
    # print(rgb_view_table)

    '''shape_view_table = pd.read_csv('tabela_tratada.txt', usecols=shape_view_fields)
    # print(shape_view_table)

    rgb_view_table = pd.read_csv('tabela_tratada.txt', usecols=rgb_view_fields)
    # print(rgb_view_table)'''

    print('--- views lidas ---')

    # divindo os dados em vaidacao  e treinamento
    # shape view
    shape_training = []
    shape_test = []
    divide_sets(shape_view_table, shape_training, shape_test)

    # rgb view
    rgb_training = []
    rgb_test = []
    divide_sets(rgb_view_table, rgb_training, rgb_test)


    # cityblock_distance(shape_view_table, rgb_view_table)


def normalize_table(values):
    scaler = proc.MinMaxScaler()

    valores_normalizados = scaler.fit_transform(values)

    return pd.DataFrame(valores_normalizados)


def divide_sets(dataset, trainingSet, testSet):
    print(dataset[1])
    for x in range(len(dataset) - 1):
        if random.random() < 0.66:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])
    return trainingSet, testSet


def cityblock_distance(dataset, att):
    distance = []
    for x in range(len(dataset)):
        for y in range(att):
            distance.append(math.sqrt(scipy.spatial.distance.cityblock(dataset[x], dataset[y])))
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


main()
