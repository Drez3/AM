import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as proc

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
	
main()
	
