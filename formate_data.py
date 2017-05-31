#least X = 7547902 ; max X = 7762614 ; diff = 214712
#least Y = 602723 ; max Y = 787862 ; diff = 185139

import xlrd
import numpy

def formate_data():
	workbook = xlrd.open_workbook('Data/Crimes_2012.xlsx')
	worksheet = workbook.sheet_by_index(0)

	## no of rows and column in the map
	div_x = 32
	div_y = 32
	diff_x = 214713
	diff_y = 185140
	least_x = 7547902
	least_y = 602723	
	size_x = diff_x/(div_x-1)
	size_y = diff_y/(div_y-1) 

	# worksheet.col(0)

	# days = 31 # 305
	days = 306
	# rows = 13032
	rows = worksheet.nrows
	cols = worksheet.ncols



	print "printing done"
	print "%d"%rows
	print "%d"%cols
 
	# matrix = [[[0 for col in range(div_x)] for row in range(div_y)] for x in range(days)]
	matrix = [[[[0 for col in range(div_x)] for row in range(div_y)] for a in range(1)] for x in range(days)]

	for i in range(1, rows):
		day = int(worksheet.cell_value(i, 1))
		x = worksheet.cell_value(i, 2)
		y = worksheet.cell_value(i, 3)
		c_y = int((y-least_y)/size_y)
		c_x = int((x-least_x)/size_x)
		# print "day:- %d row:- %d col:- %d"%(day, c_y, c_x)
		# matrix[day][c_y][c_x] +=1
		matrix[day][0][c_y][c_x] +=1
		# print "day:- %d row:- %d"%(day, c_y*div_x + c_x)
		# matrix[day][c_y*div_x + c_x] +=1

	return numpy.array(matrix);
	
# for x in range(len(matrix)):
#     print matrix[x]	

# formate_data()

# for i in xrange(1,10):
# 	pass
