import csv
import functools
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as proc
import random

def main():

	shape_view_fields = ['REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5','SHORT-LINE-DENSITY-2','VEDGE-MEAN','VEDGE-SD','HEDGE-MEAN','HEDGE-SD']
	
	rgb_view_fields = ['INTENSITY-MEAN','RAWRED-MEAN','RAWBLUE-MEAN','RAWGREEN-MEAN','EXRED-MEAN','EXBLUE-MEAN','EXGREEN-MEAN','VALUE-MEAN','SATURATION-MEAN','HUE-MEAN']
	
	shape_view_table = pd.read_csv('tabela_tratada_para_testes.txt', usecols=shape_view_fields)
	#print(shape_view_table)
	
	rgb_view_table = pd.read_csv('tabela_tratada_para_testes.txt', usecols=rgb_view_fields)
	#print(rgb_view_table)
	
	
	tabela_shape_normal = normalize_table(shape_view_table.values)
	
	tabela_rgb_normal = normalize_table(rgb_view_table.values)
	
	print('tabela shape normalizada')
	
	print(tabela_shape_normal)
	print(tabela_shape_normal.shape)
	
	print()
	
	dis_shape = create_dissimilarity_matrix(tabela_shape_normal, tabela_shape_normal.shape[0], tabela_shape_normal.shape[1])
	dis_rgb = create_dissimilarity_matrix(tabela_rgb_normal, tabela_rgb_normal.shape[0], tabela_rgb_normal.shape[1])
	
	print(dis_shape)
	
	print()
	
	print(dis_rgb)
	
	tables = []
	tables.append(dis_shape)
	tables.append(dis_rgb)
	
	MRDCA_RWG(tables, 7, 3)
	
def create_dissimilarity_matrix (table, row_nb, col_nb):

		matriz = np.empty([row_nb, row_nb])
		
		for i in range(row_nb):
			for j in range(row_nb):
				matriz[i][j] = euclidian_distance(table.iloc[[i]].values, table.iloc[[j]].values, col_nb)
		
		return matriz
	
def euclidian_distance (row1, row2, row_len):
	
	temp = 0
	for i in range(row_len):
		temp += ((row1[0][i] - row2[0][i]) ** 2)
	result = temp ** (0.5)
	return result
	
def normalize_table (values):
	
	scaler = proc.MinMaxScaler()
	
	valores_normalizados = scaler.fit_transform(values)
	
	return pd.DataFrame(valores_normalizados)
	
	
def MRDCA_RWG (tables, k, q):
	
	#vamos la, com calma
	
	t = 0
	
	#inicializando os pesos
	weights = []
	for  i in range(len(tables)):
		weights.append(1)
		
	#inicializando os prototipos
	g = []
	numbers = list(range(len(tables[0])))
	random.shuffle(numbers)
	for i in range(k):
		
		g.append(numbers.pop())
		
	print(g);
	
	#inicializando as particoes
	p = [[],[],[],[],[],[],[]]
	for table in tables:
		best_prototype = 0
		
		for i in table:
			best_dist = 1000000000
			print('outro loop')
			for j in range(k):
				if (i[g[j]] < best_dist):
					best_dist = i[g[j]]
					best_prototype = j
					p[j].append(i)
					print ('best_dist ' + str(best_dist))
	print ('best ' + str(best_prototype))
	#for i in range(tables)
	print('partitions')
	for i in range(len(p)):
		print()
		print(i)
		print(p[i])
		print()
	
	#step 1
	test = 1
	#while test != 0:
	#	t += 1
		
		#step 2
		
		#step 3
	#	test = 0
		

#def get_weight ():

# def get_best_prototype (partition, dis_matrix, ): # Eq (12)
	
	# for elem in partition:
		# result = 0
		# for matrix in dis_matrix:
			#result += 
			
# metodos auxiliares
def produtorio (fatores):
	return functools.reduce(operator.mul, fatores, 1)
	
main()
	
