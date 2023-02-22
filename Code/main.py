import numpy 
import pandas as pd
import matplotlib as mtp
import time

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

#PART 3

print('Number of NaN for each column :\n')
print(trainDataFrame.isnull().sum(axis = 0))

print('\n\nReplacing NaN values with avg of corresponding column :')
trainDataFrame['workclass'].fillna(value=trainDataFrame['workclass'].mode()[0], inplace=True)
trainDataFrame['occupation'].fillna(value=trainDataFrame['occupation'].mode()[0], inplace=True)
trainDataFrame['native-country'].fillna(value=trainDataFrame['native-country'].mode()[0], inplace=True)
# print('Number of NaN for each column :\n')
# print(trainDataFrame.isnull().sum(axis = 0))

#PART 4 

print('Deleting column(s) containing unique values (applied to __fnlwgt__) :')
trainDataFrame = trainDataFrame.drop('fnlwgt', axis=1)
print(trainDataFrame)

#PART 5

numOfFemales = (trainDataFrame['sex']==1).sum()
numOfMales = (trainDataFrame['sex']==0).sum()
print('\nNumber of females :', numOfFemales)
print('\nNumber of males :', numOfMales)

conditions = [(trainDataFrame['sex']==0) & (trainDataFrame['marital-status']=='Married-civ-spouse')]
marriedMenNum = (numpy.where(conditions, 1, 0) == 1).sum()
print('\nNumber of married men :', marriedMenNum)

#PART 6 

conditions = [(trainDataFrame['race']=='Black') & (trainDataFrame['workclass']=='Private') & (trainDataFrame['age'] >= 30)]
privateWorkingBlackMenUpper30 = (numpy.where(conditions, 1, 0)==1).sum()
print('\nNumber of black men older than 30 working private :', privateWorkingBlackMenUpper30)

#PART 7

start = time.time()
print('\nAverage working hour of people with bachelors :', trainDataFrame.loc[(trainDataFrame['education']=='Bachelors'), 'hours-per-week'].mean())
end = time.time() - start
print('Total time spent :', end)

#PART 8

totalWorkingHours=0
totalMatchedPeople=0
start = time.time()
for i in range(len(trainDataFrame)) :
    if(trainDataFrame.loc[i, 'education'] == 'Bachelors') :
        totalWorkingHours += trainDataFrame.loc[i, 'hours-per-week']
        totalMatchedPeople += 1
end=time.time()-start
if(totalMatchedPeople != 0) :
    print('\nWorking hours mean :', totalWorkingHours/totalMatchedPeople)
    print('Total time spent :', end)

#PART 9
