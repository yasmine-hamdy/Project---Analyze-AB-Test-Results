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

# Re-create new_page_converted and old_page_converted and find the  (𝑝′𝑛𝑒𝑤  -  𝑝′𝑜𝑙𝑑)  value 10,000 times using the same simulation process used above
# Store all  (𝑝′𝑛𝑒𝑤  -  𝑝′𝑜𝑙𝑑)  values in a NumPy array called `p_diffs

# Sampling distribution 
p_diffs = []

new_page_converted = np.random.binomial(n_new, p_converted_pop, 10000)/n_new
old_page_converted = np.random.binomial(n_old, p_converted_pop, 10000)/n_old
p_diffs = new_page_converted - old_page_converted

# Plot a histogram of the p_diffs
plt.hist(p_diffs);
plt.title('Difference in "Converted" Probability - Sampling Distribution')
plt.xlabel('Differences')
plt.ylabel('Number of occurences')
plt.axvline(obs_diff, c='r');

# What proportion of the p_diffs are greater than the actual difference observed in the df2 data?
(p_diffs > obs_diff).mean()
# This is the p-value, and it represents the probability of the observed change in
# average conversion occurring or an average change even more in favor of an 
# increase in conversion given there was actually no change in conversion. 
# The p-value above leads us to fail to reject the null hypothesis because 
# it's higher than the error threshhold we had set (5%).


# Using Built-in Methods for Hypothesis Testing

import statsmodels.api as sm

# number of conversions with the old_page
convert_old = df2[(df2.landing_page == 'old_page') & (df2.converted)].count()[0]

# number of conversions with the new_page
convert_new = df2[(df2.landing_page == 'new_page') & (df2.converted)].count()[0]

# number of individuals who were shown the old_page
n_old = df2[df2['landing_page'] == 'old_page'].count()[0]

# number of individuals who received new_page
n_new = df2[df2['landing_page'] == 'new_page'].count()[0]

# use sm.stats.proportions_ztest() to compute your test statistic and p-value
z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
print(z_score, p_value)

# The earlier p-value was about 0.908 and this p-value is almost 0.91 and they 
# both are less than the type I error thus supporting the previous finding of 
# failing to reject the null hypothesis. Morevoer, the z-score of 1.31 is less 
# than  𝑍𝛼  or  𝑍0.05  which is 1.645 for one-tailed tests; and since this is a 
# right-tailed test, we again fail to reject the null hypothesis.


# Part III - A regression approach 

# Since each row in the df2 data is either a conversion or no conversion, Logistic Regression should be performed in this case

#  fit the regression model 
df2['intercept']= 1
df2[['control', 'ab_page']]= pd.get_dummies(df2['group'])
df2=df2.drop('control', axis=1)
df2.head()

logit_mod =sm.Logit(df2['converted'],df2[['intercept', 'ab_page']])

results = logit_mod.fit()

# model summary
results.summary2()

# The p-value associated with ab_page in this regression model is 0.1899. It differs from the value computed in part II above due to the difference in the null and alternative hypotheses. In the previous part, this was the hypothesis test and it was one-sided:

# 𝐻0 :  𝑝𝑜𝑙𝑑  +  𝑝𝑛𝑒𝑤  >= 0

# 𝐻1  :  𝑝𝑜𝑙𝑑  +  𝑝𝑛𝑒𝑤  < 0

# In the regression model, however, the hypothesis test is different and it is two sided. It is as follows:

# 𝐻0 :  β𝑖  = 0

# 𝐻1  :  β𝑖  ≠ 0

# The current p-value is greater than a type I error rate of 0.05. The p-value suggests that there is statistical evidence that the population slope associated with the treatment group in relating to conversion is non-zero.

# The summary shows that conversion is 0.985 times as likely in the treatment group than the control group, holding all other variables constant.

# Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in

# Read the countries.csv
countries = pd.read_csv('countries.csv')
countries.head()

# Join with the df2 dataframe
df_merged = pd.merge(df2, countries, on='user_id', how='outer')
df_merged.head()

# Create the necessary dummy variables
df_merged[['UK', 'US', 'CA']] = pd.get_dummies(df_merged['country'])
df_merged.head()

# fit regression model
logit_mod =sm.Logit(df_merged['converted'],df_merged[['intercept', 'UK', 'US']])

results = logit_mod.fit()
results.summary2()

# look at an interaction between page and country to see if are there significant 
# effects on conversion.
df_merged['US_ab_page'] = df_merged['US'] * df_merged['ab_page']
df_merged['UK_ab_page'] = df_merged['UK'] * df_merged['ab_page']

# Fit your model, and summarize the results

logit_mod =sm.Logit(df_merged['converted'],df_merged[['intercept', 'ab_page', 'US', 'UK', 'US_ab_page', 'UK_ab_page']])

results = logit_mod.fit()
results.summary2()

# Again, all of coefficients' (not the intercept) p-values in the summary are 
# greater than the type I error rate (0.05). Therefore, we would choose to 
# reject the null hypotheses of the regression model because there is statistical
# evidence that the population slope associated with the specified coefficient 
# in relating to conversion is non-zero holding all others constant.

# The summary shows that after adding the interaction variables:

# Conversion is 0.98 times as likely in the treatment group than the control group, holding all other variables constant.
# Conversion is 0.99 times as likely for US residents than CA residents, holding all other variables constant.
# Conversion is 0.98 times as likely for Uk residents than CA residents, holding all other variables constant.
# Conversion is 1.03 times as likely for US residents in the treatment group than CA residents, holding all other variables constant.
# Conversion is 0.95 times as likely for UK residents in the treatment group than CA residents, holding all other variables constant.

# From my point of view, we can do without the interaction variables because they
# didn't add significant changes to our model.