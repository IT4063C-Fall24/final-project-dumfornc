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
# 
# I don't see ways to merge these specific datasets (though I may discover and merge others later) but mostly I plan to syntesize the data from multiple of these sources to create graphics by putting data on differnet topics side by side to show pros and cons of remote work.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[5]:


# Start your code here
import pandas as pd


# In[31]:


productiviy_df = pd.read_csv(r'Data_Sources\remote_work_productivity.csv')

# TWDP_df = pd.read_excel(r'Data_Sources\remote.xlsx') <- This has header rows and such that will require a lot of formatting to read in so it feels outside the scope of this checkpoint

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


# In[17]:


adv_df


# In[32]:


challenge_df


# In[21]:


productiviy_df


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->
# - https://docs.python.org/3/library/stdtypes.html
# - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html 

# In[2]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

