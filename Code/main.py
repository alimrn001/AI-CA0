import numpy 
import pandas as pd
import matplotlib as mtp

#PART 1

print('Reading CSV file into dataframe and showing contents...')
trainDataFrame = pd.read_csv(r'../train.csv')
print(trainDataFrame)

print('\n---------------------\n')

print('Showing train csv file info using dataframe method info()')
trainDataFrame.info()

print('\n---------------------\n')

print('Showing first 5 rows of train dataframe using head() method.')
print(trainDataFrame.head())

print('\n---------------------\n')

print('Showing last 5 rows of train dataframe using head() method.')
print(trainDataFrame.tail())

print('\n---------------------\n')
print('Showing a general description of the train dataframe using describe() method...')
print(trainDataFrame.describe())

# PART 2
print('\n---------------------\n')

trainDFNumericalColNames = trainDataFrame.select_dtypes(include=numpy.number).columns.to_list()
print('Columns that have numerical data :')
print(trainDFNumericalColNames)

trainDFCategorialColNames = trainDataFrame.select_dtypes(exclude=["number","bool_"]).columns.to_list()
print('Columns that have categorial data :')
print(trainDFCategorialColNames)

print('\n Changing male/female in train sataframe to 0/1 : \nnew train dataframe : \n')
trainDataFrame = trainDataFrame.replace(['Male', 'Female'] , [0, 1])
print(trainDataFrame)

print('\n---------------------\n')

#PART3

print('Number of NaN for each column :\n')
print(trainDataFrame.isnull().sum(axis = 0))

print('Replacing NaN values with avg of corresponding column :')
trainDataFrame['workclass'].fillna(value=trainDataFrame['workclass'].mode()[0], inplace=True)
trainDataFrame['occupation'].fillna(value=trainDataFrame['occupation'].mode()[0], inplace=True)
trainDataFrame['native-country'].fillna(value=trainDataFrame['native-country'].mode()[0], inplace=True)
# print('Number of NaN for each column :\n')
# print(trainDataFrame.isnull().sum(axis = 0))

#PART4 
print('Deleting column(s) containing unique values (applied to __fnlwgt__) :')
trainDataFrame = trainDataFrame.drop('fnlwgt', axis=1)
print(trainDataFrame)