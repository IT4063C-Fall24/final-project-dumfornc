#!/usr/bin/env python
# coding: utf-8

# # Effectiveness of Remote Employees📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# Many companies, esspecially in the IT sector, allowed employees to work from home during covid and some have continued the trend. It is still unclear if employers should allow their employees to work from home. I find this esspecially interesting since I find that my productivity drops drastically when I am in office but as I understand it most people are far more productive in the office than at home. 

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# Does working from home ultimately benifit employees and employeers?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# Some visualizations like bar charts to show productivity changes over time, heatmaps to explore factors influencing productivity, and pie charts for comparing remote vs. in-office work outcomes could be a part of the answer. Ultimately I think the answer to this question will be data comparing factors like mental health, stress, productivity, engagement, coworker relations, and company profit (specifically looking at costs of outfitting employees to work from home vs housing them in an office and potentially having to do both) before and after remote work to see how the numbers stack up for both sides.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->
# 3 Data Sources:
# - Teleworking during the pandemic (https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsremotexlsx?resource=download)
# - Remote Work Productivity (https://www.kaggle.com/datasets/mrsimple07/remote-work-productivity)
# - Top challenges and advantages of remote work (https://www.gallup.com/401384/indicator-hybrid-work.aspx)
# - Remote Work & Mental Health (https://www.kaggle.com/datasets/waqi786/remote-work-and-mental-health)
# - Relationship Between Remote Work and Productivity (https://www.bls.gov/opub/btn/volume-13/remote-work-productivity.htm)
# 
# I don't see ways to merge these specific datasets (though I may discover and merge others later) but mostly I plan to syntesize the data from multiple of these sources to create graphics by putting data on differnet topics side by side to show pros and cons of remote work.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[44]:


# Start your code here

# Default Imports:
import numpy as np
import pandas as pd
import plotly.express as px

# ML Imports:
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


# In[3]:


productiviy_df = pd.read_csv(r'Data_Sources\remote_work_productivity.csv')

mental_health_df = pd.read_csv(r'Data_Sources\Impact_of_Remote_Work_on_Mental_Health.csv')


# In[4]:


# This spreadsheet has several sections so to read it in python it will need to be broken into several sections using the formatt in the next cell.
#  As of right now the data in this sheet does not seem particularly usefull so I will not take the time to break it all the way out but I will leave it here
#  in case it becomes useful later.
TWDP_df = pd.read_excel(r'Data_Sources\remote.xlsx')

TWDP_df


# In[5]:


TWDP_age_df = TWDP_df[3:8]

TWDP_age_df.rename(columns={'Select characteristics': 'Age'}, inplace=True)

TWDP_age_df


# In[6]:


# This breaks up a dataset that is just lists of statements and the percentage of surveyed people who agreed
#  This breaks it down into two dataframes that are readable
advantagesDict = {}
challengesDict = {}
with open(r'Data_Sources\Top_Adv_Chlng.csv', 'r') as file:
    lineNumber = 0
    for line in file.readlines():
        lineNumber += 1
        if lineNumber in [3, 4, 5, 6, 7]:
            lineParts = line.strip().split(',')
            advantagesDict[lineParts[0]] = lineParts[1]
        elif lineNumber in [10, 11, 12, 13, 14, 15]:
            lineParts = line.strip().split(',')
            challengesDict[lineParts[0]] = lineParts[1]
        elif lineNumber == 16:
            lineParts = line.strip().split(',')
            key = f'{lineParts[0]},{lineParts[1]}'
            challengesDict[key] = lineParts[2]

adv_df_setup = {'Top Advantages': advantagesDict.keys(), 
            '% Selected as Benefits': advantagesDict.values()}
 
challenge_df_setup = {'Top Challenges': challengesDict.keys(), 
            '% Selected as Challenges': challengesDict.values()}

adv_df = pd.DataFrame(adv_df_setup)
challenge_df = pd.DataFrame(challenge_df_setup)


# In[7]:


# This is a new dataset, it has no nulls and does not appear to need cleaning
remoteProductivityRelationDf = pd.read_csv(r'Data_Sources\ProductivityandRemoteRelation.csv')
# remoteProductivityRelationDf.info()
remoteProductivityRelationDf


# In[8]:


adv_df


# In[9]:


challenge_df


# In[10]:


productiviy_df


# In[11]:


mental_health_df


# ## Visualizations

# In[12]:


# Here are some visualizations to start to get a feel for the data
productivityVizDf = productiviy_df[['Employment_Type', 'Hours_Worked_Per_Week', 'Productivity_Score', 'Well_Being_Score']].groupby(['Employment_Type'], as_index=False).mean()

productivityVizDf.plot(kind='bar', x='Employment_Type', y=['Hours_Worked_Per_Week', 'Productivity_Score', 'Well_Being_Score'], title='Productivity and well being of differnet employee modalities')


# This visualization is comparing in office employees to remote employees in three areas: hours worked per week, prodctivity rating, and wellnes rating
# Importaint insights are that the remote employees have the advantage in every area. They are working less time while being more productive and still reporting
# higher levels of wellness than in-office employees.

# In[13]:


mentalHealthVizOneDf = mental_health_df[['Work_Location', 'Work_Life_Balance_Rating']].groupby(['Work_Location'], as_index=False).mean()

# I used plotly here so that the figure was interactive and the user could scroll over to see the differences
plot = px.bar(mentalHealthVizOneDf, x='Work_Location', y='Work_Life_Balance_Rating', title='Work life balance of different employee modalities')

plot.update_layout(yaxis=dict(range=[1, 5]), width=800)

plot.show()


# This visualization is comparing the rating of work life balance for different types of employees. 
# It seems to show no significant difference in the work life balance of remote vs in person employess.

# In[14]:


mentalHealthVizTwoDf = mental_health_df

# Group data by Work_Location and Stress_Level to get counts
sunburst_data = mentalHealthVizTwoDf.groupby(['Work_Location', 'Stress_Level']).size().reset_index(name='Count')

# Create the sunburst plot
plot = px.sunburst(
    sunburst_data,
    path=['Work_Location', 'Stress_Level'],  # Define the hierarchy
    values='Count',  # Size of each slice
    color='Stress_Level',  # Color by stress level
    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}  # Custom colors
)

# Update layout for readability
plot.update_layout(
    title="Stress Level Distribution by Work Location",
    margin=dict(t=40, l=0, r=0, b=0)
)

plot.show()


# This visualization shows the the breakdown of amount of stress among the differnt work types. They are very similar again, though it is worth noting that onsite has the least "High" stressed individuals and remote has the most. This would suggest that overall it is not a huge difference but working remotly may be slightly more stressful.

# In[15]:


mentalHealthVizThreeDf = mental_health_df

# Group data by Work_Location and Productivity_Change to get counts
sunburst_data = mentalHealthVizThreeDf.groupby(['Work_Location', 'Productivity_Change']).size().reset_index(name='Count')

# Create the sunburst plot
plot = px.sunburst(
    sunburst_data,
    path=['Work_Location', 'Productivity_Change'],  # Define the hierarchy
    values='Count',  # Size of each slice
    color='Productivity_Change',  # Color by stress level
    color_discrete_map={'Decrease': 'red', 'No Change': 'grey', 'Increase': 'green'}  # Custom colors
)

# Update layout for readability
plot.update_layout(
    title="Change in productivity by Work Location",
    margin=dict(t=40, l=0, r=0, b=0)
)

plot.show()


# This visualization shows the change in productivity bewtween different work modalities. All of them had more decreases than increases but again there do not appear to be significant statistical differances between the categories. It would seem upon cursory analysis that there many not be clear trends about remote work's effects on productivity.

# The data I found seems to have already been cleaned and I did not find any issues that I needed to adress.

# I did not really recieve any prior feedback.

# ## Machine Learning Plan
# - I will try a polynomial regression with remote_work_productivity.csv to predict the productivity of employees
# - I am worried that there will not be enough data to train the model well as not all of the data seems to be particularly interlinked
# - I will experiment with several different degrees to see if any of them can get a good fit

# ## Machine Learning Implementation Process
# 

# #### Ask

# In[25]:


# I will be working with productivity_df

# The employee ID column is not going to be of use to us so lets remove it before we start doing any work with the data
productiviy_df_noID = productiviy_df.drop('Employee_ID', axis=1)

# In this info display we can see that there are no null values
print(productiviy_df_noID.info())

# This describe gives us an idea of the scale of the numerical columns, their ranges are similar and should not need much, if any, adjustment
print(productiviy_df_noID.describe())


# #### Prepare

# In[26]:


# I am not a SME but I don't see any variables that we need to split the data by so I am going to use a standard test_train_split
train_set, test_set = model_selection.train_test_split(productiviy_df_noID, test_size=.2, random_state=7)

# Lets split the training set into X and Y
productivity_X = train_set.drop('Productivity_Score', axis=1)
productivity_Y = train_set['Productivity_Score'].copy()

print(productivity_X.head())
print('-------------')
print(productivity_Y.head())


# #### Process

# In[37]:


# Start by splitting X into numerical and catagorical
productivity_train_cat = productivity_X[['Employment_Type']]
productivity_train_num = productivity_X.drop('Employment_Type', axis=1)

# As I mentioned above there are no missing values to impute so we are skipping that step

# I don't think this data should need a large amount of scalling but lets do it just in case (and to demonstrate I can)
scaler = StandardScaler()

productivity_train_num_scaled = scaler.fit_transform(productivity_train_num)
productivity_train_num_scaled = pd.DataFrame(productivity_train_num_scaled, columns=productivity_train_num.columns, index=productivity_train_num.index)

print(productivity_train_num_scaled.head())

# Now lets encode the categorical features
print(productivity_train_cat.value_counts())

cat_encoder = OneHotEncoder(sparse_output=True)
productivity_train_cat_encoded = cat_encoder.fit_transform(productivity_train_cat)

cat_encoder.categories_


# In[42]:


# Now lets do this again in pipline form for reproducability now that we know it works
num_features = productivity_train_num.columns
cat_features = productivity_train_cat.columns

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('one-hot-encoder', OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

productivity_X_prepared = full_pipeline.fit_transform(productivity_X)

all_columns = productivity_train_num.columns.tolist() + cat_encoder.categories_[0].tolist()

productivity_X_prepared = pd.DataFrame(productivity_X_prepared, columns=all_columns, index=productivity_X.index)
productivity_X_prepared.head()


# #### Analyze & Evaluate

# In[48]:


# Lets split the test data into X & Y and then clean the X
productivity_X_test = test_set.drop('Productivity_Score', axis=1)
productivity_Y_test = test_set[['Productivity_Score']].copy()

productivity_X_test_clean = full_pipeline.transform(productivity_X_test)

# To test many different degrees we will create a function that intakes the degree and then outputs a dictonary with degree and rmse
#   We can then compare to find the one with the lowest rmse
def degreeToRMSE(degree) -> dict:
    poly_reg = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree)),
        ('poly_reg', LinearRegression())
    ])

    poly_reg.fit(productivity_X_prepared, productivity_Y)

    train_prediction = poly_reg.predict(productivity_X_prepared)
    poly_train_mse = mean_squared_error(productivity_Y, train_prediction)
    poly_train_rmse = np.sqrt(poly_train_mse)

    test_prediction = poly_reg.predict(productivity_X_test_clean)
    poly_test_mse = mean_squared_error(productivity_Y_test, test_prediction)
    poly_test_rmse = np.sqrt(poly_test_mse)

    return {'train set RMSE': poly_train_rmse, 'test set RMSE': poly_test_rmse}


# In[52]:


# Now we will loop though and create a dictionary containing several potential degrees and their RMSE outputs
rmse_returns = {}
for i in range(1, 11):
    rmse_returns[i] = degreeToRMSE(i)

for key, value in rmse_returns.items():
    print(f'{key} degrees:')
    for key2, value2 in value.items():
        print(f'{key2}: {value2}')


# The model with only one degree had the best performance with the test set. 
# Degrees 2-6 all performed pretty well also, but they were slowly starting to overfit.
# After degree 7 the overfitting became obvious.
# 
# It would seem that a single degree was the best tested model.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# - https://docs.python.org/3/library/stdtypes.html
# - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html 
# - https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.plot.html
# - https://plotly.com/python/
# - ChatGPT

# In[53]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

