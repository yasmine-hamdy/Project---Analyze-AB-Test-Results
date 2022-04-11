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

# Double Check all of the incorrect rows were removed from df2 - 
# Output of the statement below should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]

# number of unique user_ids in df2
df2.user_id.nunique()

# repeated user_id in df2
df2.loc[df2.user_id.duplicated(), :]

# Display the rows for the duplicate user_id
df2.query("user_id == 773192")

# Remove one of the rows with a duplicate user_id..
# Hint: The dataframe.drop_duplicates() may not work in this case because the rows with duplicate user_id are not entirely identical. 
df2 = df2.drop(labels=1899, axis=0)
# Check again if the row with a duplicate user_id is deleted or not
df2.query("user_id == 773192")

# probability of an individual converting regardless of the page they receive
p_converted_pop = df2.converted.mean()
p_converted_pop

# Given that an individual was in the control group, what is the probability they converted
control_conv = (df2.query('group == "control"')['converted']).mean()
control_conv

# Given that an individual was in the treatment group, what is the probability they converted?
treat_conv = (df2.query('group == "treatment"')['converted']).mean()
treat_conv

# Calculate the actual difference (obs_diff) between the conversion rates for the two groups.
obs_diff = treat_conv - control_conv
obs_diff

# What is the probability that an individual received the new page?
prob_new_page = (df2[df2['landing_page'] == 'new_page'].count()[0]) / df2.shape[0]
prob_new_page


# Part II - A/B Test

# 𝐻0 : 𝑝𝑜𝑙𝑑 + 𝑝𝑛𝑒𝑤 >= 0

# 𝐻1 : 𝑝𝑜𝑙𝑑 + 𝑝𝑛𝑒𝑤 < 0

# Under the null hypothesis  𝐻0 , assume that  𝑝𝑛𝑒𝑤  and  𝑝𝑜𝑙𝑑  are equal. Furthermore, assume that  𝑝𝑛𝑒𝑤  and  𝑝𝑜𝑙𝑑  both are equal to the converted success rate in the df2 data regardless of the page. So, our assumption is:

# 𝑝𝑛𝑒𝑤  =  𝑝𝑜𝑙𝑑  =  𝑝𝑝𝑜𝑝𝑢𝑙𝑎𝑡𝑖𝑜𝑛

# In this section, we will:

# Simulate (bootstrap) sample data set for both groups, and compute the "converted" probability  𝑝  for those samples.
# Use a sample size for each group equal to the ones in the df2 data.
# Compute the difference in the "converted" probability for the two samples above.
# Perform the sampling distribution for the "difference in the converted probability" between the two simulated-samples over 10,000 iterations; and calculate an estimate.

# conversion rate for  𝑝𝑛𝑒𝑤  under the null hypothesis
p_converted_pop

# conversion rate for  𝑝𝑜𝑙𝑑  under the null hypothesis
p_converted_pop

# What is  𝑛𝑛𝑒𝑤 , the number of individuals in the treatment group?
n_new = df2[df2['landing_page'] == 'new_page'].count()[0]
n_new 

# What is  𝑛𝑜𝑙𝑑 , the number of individuals in the control group?
n_old = df2[df2['landing_page'] == 'old_page'].count()[0]
n_old

# Simulate a Sample for the treatment Group
# Simulate  𝑛𝑛𝑒𝑤  transactions with a conversion rate of  𝑝𝑛𝑒𝑤  under the null hypothesis
# hint: use np.random.choice()
new_page_converted = np.random.choice([0,1], size=(1, n_new), p=[1-p_converted_pop, p_converted_pop])
new_page_converted

# Simulate a Sample for the control Group
# Simulate  𝑛𝑜𝑙𝑑  transactions with a conversion rate of  𝑝𝑜𝑙𝑑  under the null hypothesis
old_page_converted = np.random.choice([0,1], size=(1, n_old), p=[1-p_converted_pop, p_converted_pop])
old_page_converted

# Find the difference in the "converted" probability  (𝑝′𝑛𝑒𝑤  -  𝑝′𝑜𝑙𝑑)  for the simulated samples above
new_page_converted.mean() - old_page_converted.mean()