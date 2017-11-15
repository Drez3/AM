import csv

def main():

	shape_view = []
	rgb_view = []
	normalized_v = []
	
	with open('tabela_tratada_para_testes.txt',newline='') as csvfile:

		reader = csv.reader(csvfile, delimiter=',', quotechar=';')
		row_nb = 0
		for row in reader:
			row_nb += 1
			print(row_nb)
		
		
		col_nb = 10
		
		csvfile.seek(0)
		
		for row in reader:
			print('row to be processed')
			print(row)
			entries = []
			for j in range(1,col_nb):
				entry = row[j]
				entries.append(entry)
			shape_view.append(entries)
		
			#entry = row[9:]
			#rgb_view.append(entry)
		normalized_v = normzalize_column(shape_view, 0, row_nb)
	
	print('printing from shape_view')
	print(shape_view)
	
	print('printing normalized values')
	print(normalized_v)
	
	print()
	print(row_nb)

	#print('printing rgb_view')
	#print(rgb_view)	

def normalize(v, mx, mn):
	x = ((v - mn)/(mx - mn))
	print ('x ' + str(x) + ' v ' +  str(v) + ' max ' + str(mx) + ' min ' + str(mn))
	return x
	
def normzalize_column(matrix, column, row_nb):

	max = 0.0
	min = 10000000000000000.0
	normalized_values = []
	print('going for the loop')
	for row in range(row_nb):
		tmp = float(matrix[row][column])
		if (max < tmp):
			max = tmp
		if(tmp < min):
			min = tmp
			
	for row in range(row_nb):
		tmp = float(matrix[row][column])
		normalized_values.append(normalize(tmp, max, min))
	
	return normalized_values
	
	

main()
	
