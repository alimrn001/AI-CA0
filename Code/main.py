import numpy 
import pandas as pd
import matplotlib as mtp

print('Reading CSV file into dataframe and showing contents...')
trainDataFrame = pd.read_csv(r'../train.csv')
print(trainDataFrame)

print('\n---------------------\n')

print('Showing train csv file info using dataframe method info()')
#print('information contains total number of columns, labels of column, data types of column, range index, number of non-null values in each column and total memory usage\n') #move to report
trainDataFrame.info()

print('\n---------------------\n')

print('Showing first 5 rows of train dataframe using head() method.')
#print('Passing value (N) as arguement to head() method will show the first N rows of dataframe, default value is set to 5 as shown : \n') #move to report
print(trainDataFrame.head())

print('\n---------------------\n')

print('Showing last 5 rows of train dataframe using head() method.')
#print('Passing value (N) as arguement to head() method will show the last N rows of dataframe, default value is set to 5 as shown : \n') #move to report
print(trainDataFrame.tail())

print('\n---------------------\n')

### move these to report file
# print('The describe() function computes and shows some data from dataframe as the following labels :\n')
# print('Count : Total number of non-null values of each column')
# print('Mean : Average/Mean value of each column')
# print('Std : The standard deviation for each column')
# print('Mean : Minimum value in each column')
# print('25% : 25% - The 25% percentile')
# print('50% : The 50% percentile')
# print('75% : The 75% percentile')
# print('Max : Maximum value in each column')
# print('\n The describe() method can also get some arguements')
###

print('Showing a general description of the train dataframe using describe() method...')
print(trainDataFrame.describe())

