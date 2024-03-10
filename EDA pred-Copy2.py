#!/usr/bin/env python
# coding: utf-8

# ## Importing packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from itertools import product

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


# # Data

# In[2]:


df = pd.read_csv("C:\\Users\\suhai\\OneDrive\\Desktop\\eda.csv")
df


# # Pre-processing of data

# In[3]:


pd.options.display.max_columns = None
df.head(10)


# In[4]:


df.info()


# In[5]:


np.where(df[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].sum(axis=1)== df['MntTotal'], 
         0, 1).sum()


# In[33]:


np.where(df[['MntRegularProds', 'MntGoldProds']].sum(axis=1)== df['MntTotal'], 0, 1).sum()


# In[7]:


np.where(df['Z_CostContact']== 3, 0, 1).sum()

np.where(df['Z_Revenue']== 11, 0, 1).sum()


# # Missing values

# In[8]:


df.isnull().sum().sum()


# # Duplicated observations 

# In[9]:


len(df[df.duplicated()])


# In[10]:


#data frames
cols_to_convert = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                   'AcceptedCmp2', 'Complain', 'Response', 'marital_Divorced', 'marital_Married',
                   'marital_Single', 'marital_Together', 'marital_Widow',  'education_2n Cycle', 
                   'education_Basic', 'education_Graduation',
                   'education_Master', 'education_PhD']

df[cols_to_convert] = df[cols_to_convert].astype('category')


# # Dropping duplicates and unwanted categories

# In[11]:


df = df.drop(['Z_Revenue', 'Z_CostContact'], axis = 1)
df = df.drop_duplicates(keep = 'first', ignore_index = True)


# # Merging categories

# In[12]:


df['education_Master'] = (df['education_2n Cycle'].astype('int') | df['education_Master'].astype('int')).astype('category')
df = df.drop(['education_2n Cycle'], axis = 1)


# In[13]:


#Create variables MaritalStatus, EducationStatus, Dependents and Alone_or_Couple.
df['MaritalStatus'] = df[['marital_Divorced', 'marital_Married',
                   'marital_Single', 'marital_Together', 'marital_Widow']].idxmax(axis=1).str.split('_').str[1]


# In[14]:


df['EducationStatus'] = df[['education_Basic', 'education_Graduation',
                   'education_Master', 'education_PhD']].idxmax(axis=1).str.split('_').str[1]


df['Dependents'] = df['Kidhome'] + df['Teenhome']


# In[15]:


df['Alone_or_Couple'] = np.where(df['MaritalStatus'].isin(['Single', 'Divorced', 'Widow']), 'Alone', 'Couple')


# In[16]:


#Create variabls AgeGroup, IncomeGroup, RecencyGroup and Customer_DaysGroup

limits_age = [24, 30, 35, 40, 45, 50, 55,  60, 65, 70, 75, 81]
labels_age = ['24-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64',  '65-69', '70-74', '75-81']
df['AgeGroup'] = pd.cut(df['Age'], bins=limits_age, labels=labels_age)

limits_recency = [ 0,  7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, float("inf")]
labels_recency = ['0-6', '7-13', '14-20', '21-27', '28-34', '35-41', '42-48', '49-55', '56-62', '63-69', '70-76', '77-83', '84-90', '91-97', '+98']
df['RecencyGroup'] = pd.cut(df['Recency'], bins=limits_recency, labels=labels_recency)


# In[17]:


limits_income = [     0,   5000,  10000,  15000,  20000,  25000,  30000,
        35000,  40000,  45000,  50000,  55000,  60000,  65000,
        70000,  75000,  80000,  85000, 90000,  95000, 100000,
       105000, 110000, float("inf")]
labels_income = ['0-5k', '5-10k', '10-15k', '15-20k', '20-25k', '25-30k', '30-35k',
       '35-40k', '40-45k', '45-50k', '50-55k', '55-60k', '60-65k',
       '65-70k', '70-75k', '75-80k', '80-85k', '85-90k', '90-95k',
       '95-100k', '100-105k', '105-110k', '+110k']
df['IncomeGroup'] = pd.cut(df['Income'], bins=limits_income, labels=labels_income)


# In[18]:


limits_customerdays = [2159, 2189, 2219, 2249, 2279, 2309, 2339, 2369, 2399, 2429, 2459,
       2489, 2519, 2549, 2579, 2609, 2639, 2669, 2699, 2729, 2759, 2789,
       2819, 2849, float("inf")]
labels_customerdays = ['2159-2188', '2189-2218', '2219-2248', '2249-2278', '2279-2308',
       '2309-2338', '2339-2368', '2369-2398', '2399-2428', '2429-2458',
       '2459-2488', '2489-2518', '2519-2548', '2549-2578', '2579-2608',
       '2609-2638', '2639-2668', '2669-2698', '2699-2728', '2729-2758',
       '2759-2788', '2789-2818', '2819-2848', '+2849']
df['Customer_DaysGroup'] = pd.cut(df['Customer_Days'], bins=limits_customerdays, labels=labels_customerdays)


# #  Exploratory Data Analysis (EDA)
# 

# ## Univariate Analysis

# In[19]:


df.describe()


# In[20]:


np.where(df['MntRegularProds'] < 0)[0]


# In[21]:


df = df.drop(np.where(df['MntRegularProds'] < 0)[0])
df = df.reset_index(drop=True)


# In[22]:


#Outlier detection (box plot)

df_plot = df[['Income',  'Age', 'Recency', 'Customer_Days','NumDealsPurchases', 'NumWebVisitsMonth', 'AcceptedCmpOverall',]]



rows = 1
cols = 7
fig = make_subplots(rows=rows, cols=cols)

k = 0
for i in range(rows):
    for j in range(cols):
        if k==len(df_plot.columns):
            break
        fig.add_trace(
        go.Box(y=df_plot[df_plot.columns[k]],
               name=df_plot.columns[k],
               boxmean=True
              ),
        row=(i+1), col=(j+1)
        )
        k = k + 1

fig.update_layout(
    showlegend = False,
    width=1000,
        height=500,
)


# In[23]:


df.describe()


# In[24]:


rows = 1
cols = 3
fig = make_subplots(rows=rows, cols=cols)


df_plot = df[[
    'MntTotal',
]]



for i in range(len(df_plot.columns)):
    fig.add_trace(
        go.Box(y=df_plot[df_plot.columns[i]],
               name=df_plot.columns[i],
               boxmean=True
              ),
        row=(1), col=(1)
        )

fig.update_xaxes( row=(1), col=(1))
df_plot = df[[
    'MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
]]


index_sort = np.array(df_plot[['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].mean().sort_values(ascending=False).index)


for i in range(len(df_plot.columns)):
    fig.add_trace(
        go.Box(y=df_plot[df_plot.columns[i]],
               name=df_plot.columns[i],
               boxmean=True
              ),
        row=(1), col=(2)
        )

fig.update_xaxes( row=(1), col=(2), categoryorder='array', categoryarray= index_sort)
df_plot = df[[
    'MntRegularProds', 'MntGoldProds',
]]

index_sort = np.array(df_plot[['MntRegularProds', 'MntGoldProds',]].mean().sort_values(ascending=False).index)


for i in range(len(df_plot.columns)):
    fig.add_trace(
        go.Box(y=df_plot[df_plot.columns[i]],
               name=df_plot.columns[i],
               boxmean=True
              ),
        row=(1), col=(3)
        )

fig.update_xaxes( row=(1), col=(3), categoryorder='array', categoryarray= index_sort)
fig.update_layout(
    showlegend = False,
    width=1000,
        height=500,
)


# Insights: Wine and meat products stand out as the categories with the highest spending.

# In[25]:


df_plot = df[['NumCatalogPurchases', 'NumStorePurchases',   'NumWebPurchases', ]]

rows = 1
cols = 1
fig = make_subplots(rows=rows, cols=cols)


index_sort = np.array(df_plot[['NumCatalogPurchases', 'NumStorePurchases',   'NumWebPurchases',]].mean().sort_values(ascending=False).index)

for i in range(len(df_plot.columns)):
    fig.add_trace(
        go.Box(y=df_plot[df_plot.columns[i]],
               name=df_plot.columns[i],
               boxmean=True
              ),
        row=(1), col=(1)
        )
    
fig.update_xaxes( row=(1), col=(1), categoryorder='array', categoryarray= index_sort)

fig.update_layout(
    showlegend = False,
    width=1000,
        height=500,
)


# Insights: Sales are higher in physical stores, followed by online (web), and finally through catalogs.

# In[26]:



df_plot = df[['IncomeGroup', 'AgeGroup', 'EducationStatus', 'MaritalStatus', 'Alone_or_Couple',
              'Kidhome', 'Teenhome', 'Dependents', 'RecencyGroup', 'NumDealsPurchases',
              'Customer_DaysGroup', 'Complain', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
              'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3','AcceptedCmp4', 'AcceptedCmp5',
              'AcceptedCmpOverall', 'Response']]



col_names = df_plot.columns
rows = 5
cols = 5
fig = make_subplots(rows=rows,
                    cols=cols,
                    specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True, }],
                           [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True, }],
                           [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True, }],
                           [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True, }],
                           [{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True, }]])

k = 0
for i in range(rows):
    for j in range(cols):
        if k==(len(col_names)):
            break
            
        Var = col_names[k]        
        if df_plot[Var].dtypes != 'category':
            df_plot = df_plot.copy()
            df_plot[Var] = df_plot[Var].astype('category')
          
        
        fig.add_trace(
            go.Histogram(
                x=df_plot[Var],
                name=col_names[k],
                histnorm = 'percent',
                hoverinfo = 'none',
                opacity = 0,
                showlegend = False
            ), row=(i+1), col=(j+1),
            secondary_y = True,
        )
        
            
        fig.add_trace(
            go.Histogram(
                x=df_plot[Var],
                name=col_names[k],
                marker_line=dict(width=1, color='black'),
            ), row=(i+1), col=(j+1)
        )
        
        
        
        if Var=='AgeGroup':
            fig.update_xaxes(title_text=Var, row=(i+1), col=(j+1), categoryorder='array', categoryarray= labels_age)
        elif Var=='RecencyGroup':
            fig.update_xaxes(title_text=Var, row=(i+1), col=(j+1), categoryorder='array', categoryarray= labels_recency)
        elif Var=='IncomeGroup':
            fig.update_xaxes(title_text=Var, row=(i+1), col=(j+1), categoryorder='array', categoryarray= labels_income)
        else:
            fig.update_xaxes(title_text=Var, row=(i+1), col=(j+1))
            
        fig.update_yaxes(title_text='count', row=(i+1), col=(j+1), secondary_y = False)
        fig.update_yaxes(title_text='%', row=(i+1), col=(j+1), secondary_y = True)
        k = k +1

fig.update_layout(
    showlegend = False,
    width=1500,
    height=1800,
    barmode = 'overlay',
)

fig.show()


# 
# *Insights*
# 
# Income: Income is an important factor for market analysis. It was observed that the distribution has an approximately normal profile centered around the mean. Thus, most people have income ranging from 3000 to 7000.
# 
# Age: Similar to income, the distribution is approximately normal. Most people are in the age range of 40 to 60 years.
# 
# Education: Most consumers have higher education (graduate, master's, and Ph.D.). The number of consumers with only basic education is low.
# 
# Marital Status: The most common marital status among consumers is married, followed by "together" and single. There are more consumers in relationships than singles.
# 
# Dependents: The presence of dependents, whether children or teenagers, is prevalent among consumers. We can analyze the influence of these groups separately.
# 
# Consumer Behavior
# Days Since Last Purchase: Consumers have a homogeneous behavior in relation to this factor.
# 
# Deals: An exponential behavior was observed regarding deals. Many customers take advantage of only one deals but do not accept others.
# 
# Customer Days: Consumers have a homogeneous behavior in relation to this factor.
# 
# Complaints: The vast majority of customers did not complain in the last two years, suggesting that we can ignore this variable in our analyses.
# 
# Purchase Location: There is a trend towards making more purchases online and in physical stores. The number of consumers who have never made a purchase through catalogs is very high .
# 
# 

# In[27]:


AcceptedCmp = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']].astype('int')
SuccessRateAcceptedCmp = AcceptedCmp.sum(axis = 0)/len(AcceptedCmp) 
SuccessRateAcceptedCmp = pd.DataFrame(SuccessRateAcceptedCmp,  columns = ['SuccessRate'])

fig = px.bar(SuccessRateAcceptedCmp*100, y='SuccessRate', color = SuccessRateAcceptedCmp.index, color_discrete_sequence=px.colors.qualitative.Plotly)

fig.update_layout(
    showlegend = False,
    xaxis_title='Campaign',
    
    yaxis_title='Success Rate (%)',
    xaxis=dict(tickangle=45),
    width=800,
    height=600,
)


# Campaign Success: The pilot campaign had a higher success rate than the others, indicating a qualitative change in this campaign compared to the others. A graph showing the success rate of the campaigns shows that campaigns 1, 3, 4, and 5 have similar results, while campaign 2 had much lower success, suggesting a change in marketing strategy.
# 

# Insights
# 
# Income: The peak of the distribution has shifted to the right, indicating that customers with higher income tend to accept the campaign.
# 
# Age: Although the distribution has not changed much, we can see that the most responsive age group is 45 to 49 years old, accounting for almost 20% of positive responses.
# 
# Education: Those with a graduate level of education have a better response, reflecting the demographics of the sample. Those with a master's level of education respond slightly worse than those with a Ph.D.
# 
# Marital Status: Singles and married individuals have the best response, but this evens out when comparing individuals who are alone and couples.
# 
# Dependents: People without children respond better to the campaign.
# 
# Consumer Behavior
# Days Since Last Purchase: The more recent the last purchase, the better the customer's responsiveness.
# 
# Deals: Customers who bought a deal prevail, but this behavior seems to reflect the exponential behavior of the previous chart.
# 
# Customer Days: Older customers tend to be more responsive.
# 
# Purchase Location: Despite catalog purchases not having as much adoption, we have a considerable number of customers who have already purchased via catalog and responded well to the campaign, indicating a more responsive behavior from these customers.
# 
# Success of Previous Campaigns: There is a trend among customers who accepted previous campaigns to accept the new one.
# 

# ## Bivariate Analysis

# In[29]:


#Correlation between the variables
df_plot = df[['Income', 'Age', 'Kidhome','Teenhome',  'Dependents', 'MntTotal', 'MntWines', 'MntFruits',
              'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntRegularProds',
              'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases',
              'NumStorePurchases', 'NumWebVisitsMonth', 'Customer_Days', 'Recency', 'AcceptedCmpOverall' ]]
corr = df_plot.corr()

fig = px.imshow(corr,
                x=corr.columns,
                y=corr.columns,
                color_continuous_scale='RdBu_r',  # Escolha sua escala de cores
                zmin = -1,
                zmax=1,
                text_auto=True)

fig.update_layout(width=1100, height=1100)
fig.update_traces(texttemplate="%{z:.2f}")

fig.show()


# Insights
# 
# 
# We observe a negative correlation between the number of children and the total consumption of products, while the number of teenagers does not show a significant correlation.
# 
# Additionally, it is interesting to note that the number of children is also negatively correlated with income. This may seem contradictory since the general intuition would be that families with more children would have higher spending demands, which could drive the pursuit of higher incomes. These trends can be visualized in the box plot and scatter plot below.
# 
# The number of dependents does not seem to affect the preference for consumed items.
# 
# 

# In[30]:


df_plot = df[['AcceptedCmpOverall', 'EducationStatus', 'MaritalStatus', 'Alone_or_Couple', 'MntTotal']]

Target = 'MntTotal'

col_names = df_plot.columns
rows = 2
cols = 2
fig = make_subplots(rows=rows, cols=cols)

k = 0
for i in range(rows):
    for j in range(cols):
        if k==(len(col_names) - 1):
            break
            
        Var = col_names[k]        
        if df_plot[Var].dtypes != 'category':
            df_plot = df_plot.copy()
            df_plot[Var] = df_plot[Var].astype('category')
        

        fig.add_trace(
            go.Box(x=df_plot[Var],
                   y=df_plot['MntTotal'],
                   name=Var,
                   showlegend = False,
                   boxmean = True,
                  ),row=(i+1), col=(j+1))
        
        fig.update_xaxes(title_text=Var, row=(i+1), col=(j+1))
        fig.update_yaxes(title_text='MntTotal', row=(i+1), col=(j+1))

        k = k +1
        

fig.update_layout(
    legend=dict(orientation="h"),
    width=1500,
    height=800,
)

fig.show()


# Insights
# 
# An analysis of consumption in relation to categorical variables, which are not included in the correlation matrix, reveals some interesting findings:
# 
# Customers who accepted previous campaigns tend to spend more.
# Individuals with higher levels of education also tend to spend more. Note that individuals with lower incomes are almost exclusively found in the basic education category.
# Surprisingly, marital status does not seem to have a significant influence on people's spending.
# 

# In[31]:


df_plot = df[['NumCatalogPurchases', 'NumStorePurchases', 'NumWebPurchases','IncomeGroup']]



col_names = df_plot.columns
rows = 2
cols = 2
fig = make_subplots(rows=rows, cols=cols)

k = 0
for i in range(rows):
    for j in range(cols):
        if k==(len(col_names) - 1):
            break
            
        Var = col_names[k]        
        if df_plot[Var].dtypes != 'category':
            df_plot = df_plot.copy()
            df_plot[Var] = df_plot[Var].astype('category')        

        fig.add_trace(
            go.Box(x=df_plot['IncomeGroup'],
                   y=df_plot[Var],
                   name=Var,
                   boxmean = True, 
                  ),row=(i+1), col=(j+1)
            )
            

        
        fig.update_xaxes(title_text='IncomeGroup', row=(i+1), col=(j+1), categoryorder='array', categoryarray= labels_income)
        fig.update_yaxes(title_text=Var, row=(i+1), col=(j+1))

        k = k +1

        
fig.update_layout(
    showlegend = False,
    width=1200,
    height=1200,
)

fig.show()


# Purchase Location
# 
# Insights
# 
# We observe that the number of purchases varies according to the location and increases as income grows. This behavior is expected since those who earn more tend to spend more.
# The number of web page visits decreases as income increases.
# Additionally, we notice that the number of web page visits initially decreases with the increase in the number of web purchases, followed by a subsequent increase.
# There is no significant change in this behavior with age.
# 

#  Recommendations
# About Campaign Responsiveness
# Customers who possess at least one of the following characteristics tend to accept the campaign better:
# 
# Higher income
# Older customers
# Those who made the last purchase more recently
# Customers without dependents (children or teenagers)
# Those who make catalog purchases
# Those who accepted previous campaigns
# The best-selling products overall are wine and meat (i.e., the average customer spent more on these items).
# 
# The most frequent purchase locations are physical stores and the web.
# 
# Action Suggested: Focus advertising campaigns on customers with at least one of the characteristics mentioned above and on the best-selling items, regardless of the sales channel.
# 
# About Customer Segmentation
# Customers with children tend to spend less.
# 
# Action Suggested: The company can create customized product packages for families with children to increase consumption among those with children.
# 
# There is a complex behavior among customers who make more online purchases and visit the web page more often.
# 
# Action Suggested: The company can conduct additional analyses to better understand this relationship and optimize the online experience.
# 
# The channel with lower overall performance is catalog purchases. However, there is a high responsiveness to the campaign among customers who make catalog purchases.
# 
# Action Suggested: Seek ways to boost catalog sales.
