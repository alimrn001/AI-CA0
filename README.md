# AI-CA0
AI spring 1401-1402 Project #0

## installing packages

### pandas 
```bash
pip install pandas
```
### matplotlib
```bash
pip install -U matplotlib
```
## questions

### Part 1

1. info() method

The info() method shows some genral information about the dataframe. 
information contains total number of columns, labels of column, data types of column, range index, number of non-null values in each column and total memory usage.

2. head() method

The head() method Shows first 5 rows of the dataframe.
Passing value (N) as argument to head() method will show the first N rows of dataframe, default value is set to 5.

3. tail() method

The head() method Shows last 5 rows of the dataframe.
Passing value (N) as argument to head() method will show the last N rows of dataframe, default value is set to 5.

4. describe() method

The describe() method Shows a general description of the train dataframe.
The describe() function computes and shows some data from dataframe as the following labels :
Count : Total number of non-null values of each column
Mean : Average/Mean value of each column')
Std : The standard deviation for each column')
Min : Minimum value in each column')
25% : 25% - The 25% percentile')
50% : The 50% percentile')
75% : The 75% percentile')
Max : Maximum value in each column')

The describe() method can also get some arguements as follows : 
```python
dataframe.describe(percentiles, include, exclude, datetime_is_numeric)
```

| Parameter | value | Description |
| --- | --- | --- |
| percentile | numbers between 0 and 1 | Optional, a list of percentiles to include in the result, default is [.25, .50, .75] |
| include | None, 'all', datatypes | Optional, a list of the data types to allow in the result |
| exclude | None, 'all', datatypes | Optional, a list of the data types to disallow in the result |
| datetime_is_numeric | True, False | Optional, default False. Set to True to treat datetime data as numeric |
