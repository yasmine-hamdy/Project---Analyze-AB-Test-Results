# Introduction
# A/B tests are very commonly performed by data analysts and data scientists. For this project, you will be working to understand the results of an A/B test run by an e-commerce website. Your goal is to work through this notebook to help the company understand if they should:

# Implement the new webpage,
# Keep the old webpage, or
# Perhaps run the experiment longer to make their decision.

# Part I - Probability
# import libraries

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)

# Read in the dataset from the ab_data.csv file and take a look at the top few rows
df = pd.read_csv('ab_data.csv')
df.head()

# number of rows in the dataset
df.shape[0]

# number of unique users in the dataset
df.user_id.nunique()

# proportion of users converted
df.converted.mean()

# number of times when the "group" is treatment but "landing_page" is not a new_page
df.groupby('group')['landing_page'].value_counts()

# missing values
df.isnull().any(axis=1).sum()

# Remove the inaccurate rows, and store the result in a new dataframe df2
df2 = df.drop(df[(df.group == 'control') & (df.landing_page == 'new_page')].index)
df2.head()
df2.drop(df2[(df2.group == 'treatment') & (df2.landing_page == 'old_page')].index, inplace=True)
df2.head()