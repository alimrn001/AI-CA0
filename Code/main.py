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
trainDataFrame['workclass'].fillna(value=trainDataFrame['workclass'].mean(), inplace=True)
trainDataFrame['occupation'].fillna(value=trainDataFrame['occupation'].mean(), inplace=True)
trainDataFrame['native-country'].fillna(value=trainDataFrame['native-country'].mean(), inplace=True)

# print('age :' , trainDataFrame['age'].isna().sum())
# print('workclass :', trainDataFrame['workclass'].isna().sum())
# print('fnlwgt :', trainDataFrame['fnlwgt'].isna().sum())
# print('education :', trainDataFrame['education'].isna().sum())
# print('education-num :', trainDataFrame['education-num'].isna().sum())
# print('marital-status :', trainDataFrame['marital-status'].isna().sum())
# print('occupation :', trainDataFrame['occupation'].isna().sum())
# print('relationship :', trainDataFrame['relationship'].isna().sum())
# print('race :', trainDataFrame['race'].isna().sum())
# print('sex :', trainDataFrame['sex'].isna().sum())
# print('capital-gain :', trainDataFrame['capital-gain'].isna().sum())
# print('capital-loss :', trainDataFrame['capital-loss'].isna().sum())
# print('hours-per-week :', trainDataFrame['hours-per-week'].isna().sum())
# print('native-country :', trainDataFrame['native-country'].isna().sum())
# print('occupation :', trainDataFrame['occupation'].isna().sum())

