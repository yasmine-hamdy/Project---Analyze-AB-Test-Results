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
