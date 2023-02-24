import numpy 
import pandas as pd
import matplotlib as mtp
import time
import matplotlib.pyplot as plt
import scipy.stats as scp

def getColumnNameByIndex(index) :
    if(index==0) :
        return 'age'
    elif(index==1) :
        return 'workclass'
    elif(index==2) :
        return 'fnlwgt'
    elif(index==3) :
        return 'education'
    elif(index==4) :
        return 'education-num'
    elif(index==5) :
        return 'marital-status'
    elif(index==6) :
        return 'occupation'
    elif(index==7) :
        return 'relationship'
    elif(index==8) :
        return 'race'
    elif(index==9) :
        return 'sex'
    elif(index==10) :
        return 'capital-gain'
    elif(index==11) :
        return 'capital-loss'
    elif(index==12) :
        return 'hours-per-week'
    elif(index==13) :
        return 'native-country'
    elif(index==14) :
        return 'salary'
    else :
        return 'index-error'

#PART 1

print('Reading CSV file into dataframe and showing contents...')
trainDataFrame = pd.read_csv(r'../train.csv')
print(trainDataFrame)

print('\n\nShowing train csv file info using dataframe method info()')
trainDataFrame.info()

print('\n\nShowing first 5 rows of train dataframe using head() method.')
print(trainDataFrame.head())

print('\n\nShowing last 5 rows of train dataframe using head() method.')
print(trainDataFrame.tail())

print('\n\nShowing a general description of the train dataframe using describe() method...')
print(trainDataFrame.describe())

# PART 2

trainDFNumericalColNames = trainDataFrame.select_dtypes(include=numpy.number).columns.to_list()
print('\n\nColumns that have numerical data :')
print(trainDFNumericalColNames)

trainDFCategorialColNames = trainDataFrame.select_dtypes(exclude=["number","bool_"]).columns.to_list()
print('Columns that have categorial data :')
print(trainDFCategorialColNames)

print('\n Changing male/female in train dataframe to 0/1 : \nnew train dataframe : \n')
trainDataFrame = trainDataFrame.replace(['Male', 'Female'] , [0, 1])
print(trainDataFrame)

#PART 3

print('Number of NaN for each column :\n')
print(trainDataFrame.isnull().sum(axis = 0))

print('\n\nReplacing NaN values with avg of corresponding column :')
trainDataFrame['workclass'].fillna(value=trainDataFrame['workclass'].mode()[0], inplace=True)
trainDataFrame['occupation'].fillna(value=trainDataFrame['occupation'].mode()[0], inplace=True)
trainDataFrame['native-country'].fillna(value=trainDataFrame['native-country'].mode()[0], inplace=True)

#PART 4 

# for i in range(0,15) :
#     print('unique stat for :', getColumnNameByIndex(i), ': ', trainDataFrame[getColumnNameByIndex(i)].is_unique)

#print('unique stat :', trainDataFrame['fnlwgt'].is_unique)

# unique_count = (trainDataFrame.nunique())
# unique_count = list((unique_count[unique_count == len(trainDataFrame)]).index)
# print(unique_count, 'to be removed')

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

print('Genrating plot...\nEach windows must be closed to let program continue.') # change "show" to "ion" except the last plot to show all windows at once

for i in range(0, 15) :
    if(i!=2) :
        plt.figure()
        trainDataFrame[getColumnNameByIndex(i)].hist(legend=True)
        plt.show()

#PART 10

trainDataFrame['age'] = ((trainDataFrame['age']-(trainDataFrame['age'].mean())) / (trainDataFrame['age'].std()))
trainDataFrame['education-num'] = ((trainDataFrame['education-num']-(trainDataFrame['education-num'].mean())) / (trainDataFrame['education-num'].std()))
trainDataFrame['capital-gain'] = ((trainDataFrame['capital-gain']-(trainDataFrame['capital-gain'].mean())) / (trainDataFrame['capital-gain'].std()))
trainDataFrame['capital-loss'] = ((trainDataFrame['capital-loss']-(trainDataFrame['capital-loss'].mean())) / (trainDataFrame['capital-loss'].std()))
trainDataFrame['hours-per-week'] = ((trainDataFrame['hours-per-week']-(trainDataFrame['hours-per-week'].mean())) / (trainDataFrame['hours-per-week'].std()))
trainDataFrame['sex'] = ((trainDataFrame['sex']-(trainDataFrame['sex'].mean())) / (trainDataFrame['sex'].std()))

print('\n\n', trainDataFrame.head())

#PART 11 

numericalDataColumns = list((trainDataFrame.select_dtypes(include=numpy.number)).columns)
#numericalDataColumns.remove('sex')

for colName in numericalDataColumns :
    salaryHIGH50K = trainDataFrame.loc[(trainDataFrame['salary']=='>50K'), colName]
    salaryLOW50K = trainDataFrame.loc[(trainDataFrame['salary']=='<=50K'), colName]
    plt.title(colName)
    plt.scatter(salaryHIGH50K, scp.norm.pdf(salaryHIGH50K, salaryHIGH50K.mean(), salaryHIGH50K.std()), label='>50K')
    plt.scatter(salaryLOW50K, scp.norm.pdf(salaryLOW50K, salaryLOW50K.mean(), salaryLOW50K.std()), label='<=50K')
    plt.legend(['>50K', '<=50K'])
    plt.show()

#PART 12 

testDataFrame = pd.read_csv(r'../test.csv')
testDataFrame = testDataFrame.replace(['Male', 'Female'] , [0, 1])
testDataFrame['age'] = ((testDataFrame['age']-(testDataFrame['age'].mean())) / (testDataFrame['age'].std()))
testDataFrame['education-num'] = ((testDataFrame['education-num']-(testDataFrame['education-num'].mean())) / (testDataFrame['education-num'].std()))
testDataFrame['capital-gain'] = ((testDataFrame['capital-gain']-(testDataFrame['capital-gain'].mean())) / (testDataFrame['capital-gain'].std()))
testDataFrame['capital-loss'] = ((testDataFrame['capital-loss']-(testDataFrame['capital-loss'].mean())) / (testDataFrame['capital-loss'].std()))
testDataFrame['hours-per-week'] = ((testDataFrame['hours-per-week']-(testDataFrame['hours-per-week'].mean())) / (testDataFrame['hours-per-week'].std()))
testDataFrame['sex'] = ((testDataFrame['sex']-(testDataFrame['sex'].mean())) / (testDataFrame['sex'].std()))

#selected columns => capital-gain, capital-loss, sex

def getIncomePredictionResult_SEX(sex_val) :
    if(sex_val < 1) :
        return '>50K'
    else :
        return '<=50K'

def getIncomePredictionResult_GAIN(capitalGain_val) :
    if((capitalGain_val < 3.5) and (capitalGain_val > 0.5)) :
        return '>50K'
    return '<=50K'

def getIncomePredictionResult_HPW(hoursPerWeek_Val) : 
    if((hoursPerWeek_Val > 0.4) and (hoursPerWeek_Val < 2.4)) :
        return '>50K'
    return '<=50K'

def getIncomePredictionResult_AGE(age_val) :
    if((age_val > 0) and (age < 2)) :
        return '>50K'
    return '<=50K'

def getIncomePredictionResult_EDNUM(educationNum_val) :
    if((educationNum_val > 0.5) and (educationNum_val < 2.3)) :
        return '>50K'
    return '<=50K'

testDataFrame['incomePrediction_sex'] = testDataFrame['sex'].map(getIncomePredictionResult_SEX)
testDataFrame['incomePrediction_GAIN'] = testDataFrame['capital-gain'].map(getIncomePredictionResult_GAIN)
testDataFrame['incomePrediction_HPW'] = testDataFrame['hours-per-week'].map(getIncomePredictionResult_HPW)
testDataFrame['incomePrediction_EDNUM'] = testDataFrame['education-num'].map(getIncomePredictionResult_EDNUM)
testDataFrame['incomePrediction_AGE'] = testDataFrame['age'].map(getIncomePredictionResult_EDNUM)

print(testDataFrame.head(15))


testDataFrame.to_csv(r'../salaryPrediction.csv',
columns = ['salary', 'incomePrediction_sex', 'incomePrediction_GAIN', 'incomePrediction_HPW', 'incomePrediction_EDNUM', 'incomePrediction_AGE'],
index = True)