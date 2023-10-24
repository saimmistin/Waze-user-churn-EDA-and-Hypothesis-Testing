# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# +
df = pd.read_csv('/Users/saim/Desktop/datasets/waze_dataset.csv')

df


# -

df.head(10)

df.size

df.describe()

# summary information
df.info()

# ## Data Exploration and Visualizations¶
#

# number of occurrence of a user opening the app during the month
plt.figure(figsize=(5,1))
sns.boxplot(x=df['sessions'], fliersize=1)
plt.title('sessions box plot');

# +
# number of occurrence of a user opening the app during the month (Histogram)

plt.figure(figsize=(5,3))
sns.histplot(x=df['sessions'])
median = df['sessions'].median()
plt.axvline(median, color='red', linestyle='--')
plt.text(75,1200, 'median=56.0', color='red')
plt.title('sessions box plot');
# -

# half of the observations having 56 or fewer sessions

# +
# occurrence of driving at least 1 km during the month

plt.figure(figsize=(5,1))
sns.boxplot(x=df['drives'], fliersize=1)
plt.title('drives box plot');

# -

# An occurrence of driving at least 1 km during the month

# helper function to plot histograms based on the
# format of the `sessions` histogram
def histogrammer(column_str, median_text=True, **kwargs):    # **kwargs = any keyword arguments from the sns.histplot() function
    median=round(df[column_str].median(), 1)
    plt.figure(figsize=(5,3))
    ax = sns.histplot(x=df[column_str], **kwargs)            # plot the histogram
    plt.axvline(median, color='red', linestyle='--')         # plot the median line
    if median_text==True:                                    # add median text unless set to False
        ax.text(0.25, 0.85, f'median={median}', color='red',
            ha='left', va='top', transform=ax.transAxes)
    else:
        print('Median:', median)
    plt.title(f'{column_str} histogram');


# histogram
histogrammer('drives')

# median drives is 48. However, some drivers had over 400 drives in the last month.

# +
#n_days_after_onboarding
#The number of days since a user signed up for the app

plt.figure(figsize=(5,1))
sns.boxplot(x=df['n_days_after_onboarding'], fliersize=1)
plt.title('n_days_after_onboarding box plot');

# +
#histogram plot of n_days_after_onboarding

histogrammer('n_days_after_onboarding', median_text=False)
# -

# near-zero to ~3,500 (~9.5 years).

# +
# total kilometers driven during the month

plt.figure(figsize=(5,1))
sns.boxplot(x=df['driven_km_drives'], fliersize=1)
plt.title('driven_km_drives box plot');
# -

# histogram of total kilometers driven during the month
histogrammer('driven_km_drives')

# half the users driving under 3,495 kilometers

# +
# total duration driven in minutes during the month

plt.figure(figsize=(5,1))
sns.boxplot(x=df['duration_minutes_drives'], fliersize=1)
plt.title('duration_minutes_drives box plot');
# -

# histogram plot of total duration driven in minutes during the month
histogrammer('duration_minutes_drives')

# half of the users drove less than ~1,478 minutes (~25 hours), but some users clocked over 250 hours over the month

# number of days the user opens the app during the month
plt.figure(figsize=(5,1))
sns.boxplot(x=df['activity_days'], fliersize=1)
plt.title('activity_days box plot');

# +
# number of days the user opens the app during the month

histogrammer('activity_days', median_text=False, discrete=True)
# -

# within the last month, users opened the app a median of 16 times. histogram shows a nearly ~500 people opening the app on each count of days. 
# however, there are ~250 people who didn't open the app at all. Also ~250 people who opened the app every day of the month.

# number of days the user drives at least 1 km during the month
plt.figure(figsize=(5,1))
sns.boxplot(x=df['driving_days'], fliersize=1)
plt.title('driving_days box plot');

# histogram of number of days the user drives at least 1 km during the month
histogrammer('driving_days', median_text=False, discrete=True)

#The type of device a user starts a session with
# pie chart 
fig = plt.figure(figsize=(3,3))
data=df['device'].value_counts()
plt.pie(data,
        labels=[f'{data.index[0]}: {data.values[0]}',
                f'{data.index[1]}: {data.values[1]}'],
        autopct='%1.1f%%'
        )
plt.title('Users by device');

# There are nearly twice as many iPhone users as Android users represented in this data

# +
#Binary target variable (“retained” vs “churned”) for if a user has churned anytime during the course of the month
#This is also a categorical variable, and as such would not be plotted as a box plot. Plot a pie chart instead.

fig = plt.figure(figsize=(3,3))
data=df['label'].value_counts()
plt.pie(data,
        labels=[f'{data.index[0]}: {data.values[0]}',
                f'{data.index[1]}: {data.values[1]}'],
        autopct='%1.1f%%'
        )
plt.title('Count of retained vs. churned');
# -

# Less than 18% of the users churned.

# +
# shows how many iPhone users were retained/churned and how many Android users were retained/churned.

# retention by device

# histogram plot
plt.figure(figsize=(5,4))
sns.histplot(data=df,
             x='device',
             hue='label',
             multiple='dodge',
             shrink=0.9
             )
plt.title('Retention by device histogram');
# -

# The proportion of churned users to retained users is consistent between device types.

# +
# churn rate per number of driving days histogram
# histogram plot, it represents the churn rate for each number of driving days

plt.figure(figsize=(12,5))
sns.histplot(data=df,
             x='driving_days',
             bins=range(1,32),
             hue='label',
             multiple='fill',
             discrete=True)
plt.ylabel('%', rotation=0)
plt.title('Churn rate per driving day');
# -

# The more times they used the app, the less likely they were to churn. While 40% of the users who didn't use the app at all last month churned.
# If people who used the app a lot churned, it would likely indicate dissatisfaction.
# When people who don't use the app churn, it might be the result of dissatisfaction in the past,
#

# 1)Analysis revealed that the overall churn rate is ~17%, and that this rate is consistent between iPhone users and Android users.
# 2)Number of driving days had a negative correlation with churn. Users who drove more days of the last month were less likely to churn.

# ## Data exploration and hypothesis testing
#

# +
# device is a categorical variable with the labels iPhone and Android
# assigns a 1 for an iPhone user and a 2 for Android
# creating a new variable


# create `map_dictionary`
map_dictionary = {'Android': 2, 'iPhone': 1}

# create new `device_type` column
df['device_type'] = df['device']

# map the new column to the dictionary
df['device_type'] = df['device_type'].map(map_dictionary)

df['device_type'].head()
# -

# average number of drives for each device type
df.groupby('device_type')['drives'].mean()

# It appears that drivers who use an iPhone device to interact with the application have a higher number of drives on average. 
# However, this difference might arise from random sampling, rather than being a true difference in the number of drives.
# To understand whether the difference is statistically significant, I can conduct a hypothesis test.

# ## Hypothesis testing
#

# I am going to conduct a two-sample t-test. Recall the steps for conducting a hypothesis test:
#
# 1)State the null hypothesis and the alternative hypothesis
#
# Hypothesis A == There is no difference in average number of drives between drivers who use iPhone devices and drivers who use Androids.
#
# Hypothesis A == There is a difference in average number of drives between drivers who use iPhone devices and drivers who use Androids.
#
# 2)Choose a signficance level
#
# 5% as the significance level and proceed with a two-sample t-test.
#
# 3)Find the p-value
# 4)Reject or fail to reject the null hypothesis

#import library for hypothesis testing
from scipy import stats

# +
# isolate the `drives` column for iPhone users.
iPhone = df[df['device_type'] == 1]['drives']

# isolate the `drives` column for Android users.
Android = df[df['device_type'] == 2]['drives']

# Perform the t-test
stats.ttest_ind(a=iPhone, b=Android, equal_var=False)
# -

# P-value is larger than the chosen significance level (5%), we fail to reject the null hypothesis. 
# We conclude that there is not a statically significant difference in the average number of drives between drivers who use iPhones and drivers who use Androids.


