import csv
import functools
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as proc
import random

def main():

	#separando as duas views
	print('--- inciando ---')
	shape_view_fields = ['REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5','SHORT-LINE-DENSITY-2','VEDGE-MEAN','VEDGE-SD','HEDGE-MEAN','HEDGE-SD']
	
	rgb_view_fields = ['INTENSITY-MEAN','RAWRED-MEAN','RAWBLUE-MEAN','RAWGREEN-MEAN','EXRED-MEAN','EXBLUE-MEAN','EXGREEN-MEAN','VALUE-MEAN','SATURATION-MEAN','HUE-MEAN']
	
	'''shape_view_table = pd.read_csv('tabela_tratada_para_testes.txt', usecols=shape_view_fields)
	#print(shape_view_table)
	
	rgb_view_table = pd.read_csv('tabela_tratada_para_testes.txt', usecols=rgb_view_fields)
	#print(rgb_view_table)'''
	
	shape_view_table = pd.read_csv('tabela_tratada.txt', usecols=shape_view_fields)
	#print(shape_view_table)
	
	rgb_view_table = pd.read_csv('tabela_tratada.txt', usecols=rgb_view_fields)
	#print(rgb_view_table)
	
	print('--- views lidas ---')
	
	#normalizando os valores das views
	tabela_shape_normal = normalize_table(shape_view_table.values)
	
	tabela_rgb_normal = normalize_table(rgb_view_table.values)
	
	print('--- views normalizadas ---')
	
	#criando as matrizes de dissimilaridade
	print('--- gerando matrizes de dissimilaridade ---')
	print('--- 1 ---')
	dis_shape = create_dissimilarity_matrix(tabela_shape_normal, tabela_shape_normal.shape[0], tabela_shape_normal.shape[1])
	print('--- 1 concluida ---')
	print('--- 2 ---')
	dis_rgb = create_dissimilarity_matrix(tabela_rgb_normal, tabela_rgb_normal.shape[0], tabela_rgb_normal.shape[1])
	print('--- 2 concluida ---')
	
	tables = []
	tables.append(dis_shape)
	#tables.append(dis_rgb)
	
	#rodando o algoritmo
	print('rodando o algoritmo')
	MRDCA_RWG(tables, 7, 3)
		
def MRDCA_RWG (tables, k, q):
	
	#vamos la, com calma
	
	t = 0
	
	#inicializando os pesos
	weights = []
	for  i in range(len(tables)):
		weights.append(1)
		
	#inicializando os prototipos
	#g = [0,1,2,3,4,5,6]
	g = []
	numbers = list(range(len(tables[0])))
	random.shuffle(numbers)
	for i in range(k):
		
		g.append(numbers.pop())
		
	print('prototipos')	
	print(g)
	print()
	
	#inicializando as particoes
	p = [[],[],[],[],[],[],[]]
	elem_p_list = []
	for table in tables:
		best_prototype = 0
		cur_row = 0
		for row in table:
			best_dist = 1000000000
			row_2_append = -1
			#print('outro loop')
			for j in range(k):
				if (row[g[j]] < best_dist):
					best_dist = row[g[j]]
					best_prototype = j
					row_2_append = cur_row
					#print ('new best_dist ' + str(best_dist))
					#print ('new best partition ' +  str(j))
			p[best_prototype].append(row_2_append)
			elem_p_list.append(best_prototype)
			cur_row += 1
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
	while test != 0:
		t += 1
		tmp_index = 0
		for prot in g:
			g[tmp_index] = compute_best_prototype(tables, p[tmp_index])
			tmp_index += 1
		print('novos prototipos')
		print(g)
		print()
		#step 2
		
		#step 3
		test = 0
		old_elemm_p_list = elem_p_list
		elem_p_list = []
		p = [[],[],[],[],[],[],[]]
		for table in tables:
			best_prototype = 0
			cur_row = 0
			for row in table:
				best_dist = 1000000000
				row_2_append = -1
				#print('outro loop')
				for j in range(k):
					if (row[g[j]] < best_dist):
						best_dist = row[g[j]]
						best_prototype = j
						row_2_append = cur_row
						#print ('new best_dist ' + str(best_dist))
						#print ('new best partition ' +  str(j))
				p[best_prototype].append(row_2_append)
				elem_p_list.append(best_prototype)
				if (best_prototype != old_elemm_p_list[cur_row]):
					test = 1
				cur_row += 1
		print(p)

#def get_weight ():

# def get_best_prototype (partition, dis_matrix, ): # Eq (12)
	
	# for elem in partition:
		# result = 0
		# for matrix in dis_matrix:
			#result += 
			
# metodos auxiliares
def produtorio (fatores):
	return functools.reduce(operator.mul, fatores, 1)

def compute_best_prototype (tabela, partition):
	best_sum = 100000000000000
	new_prototype = 0
	overall_elem = 0
	
	for elem in tabela[0]:
		dissim_sum = 0
		for element in range(len(partition)-1):
			dissim_sum += elem[element]
			if(dissim_sum >= best_sum):
				break
			
		if(dissim_sum < best_sum):
			best_sum = dissim_sum
			new_prototype = overall_elem
		overall_elem += 1
	return new_prototype
def create_dissimilarity_matrix (table, row_nb, col_nb):

	matriz = np.empty([row_nb, row_nb])
	
	cur_row_nb = 0
	for i in range(row_nb):
		cur_row_nb += 1
		print('gerando --- ' + str(((cur_row_nb * 100) / row_nb)) + '% (' + str(cur_row_nb) + '/' + str(row_nb) + ')')
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
	
main()