#!/usr/bin/env python
# coding: utf-8

# # Real Estate

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # 1. Import Data

# In[2]:


df_train = pd.read_csv('train.csv')
df_train.head()


# In[3]:


df_test = pd.read_csv('test.csv')
df_test.head()


# In[4]:


df_train.columns


# In[5]:


df_test.columns


# In[6]:


len(df_train)


# In[7]:


len(df_test)


# In[8]:


df_train.describe()


# In[9]:


df_test.describe()


# In[10]:


df_train.info()


# In[11]:


df_test.info()


# # 2. Figure out the primary key and look for the requirement of indexing.

# UID is unique userID value in the train and test dataset. So an index can be created from the UID feature

# In[12]:


#set the DataFrame index using existing columns.
df_train.set_index(keys=['UID'], inplace=True)
df_test.set_index(keys=['UID'], inplace=True)


# In[13]:


df_train.head(5)


# In[14]:


df_test.head(5)


# 3. Gauge the fill rate of the variables and devise plans for missing value treatment. Please explain explicitly the reason for the treatment chosen for each variable.

# In[15]:


#percentage of missing values in training set
missing_list_train=df_train.isnull().sum()*100/len(df_train)
missing_values_df_train=pd.DataFrame(missing_list_train,columns=['Percentage of missing values'])
missing_values_df_train.sort_values(by=['Percentage of missing values'],inplace=True,ascending=False)
missing_values_df_train[missing_values_df_train['Percentage of missing values']>0][:10]
#BLOCKID can be dropped, since it is 100%missing values


# In[16]:


#percentage of missing values in testing set
missing_list_test=df_test.isnull().sum()*100/len(df_train)
missing_values_df_test=pd.DataFrame(missing_list_test,columns=['Percentage of missing values'])
missing_values_df_test.sort_values(by=['Percentage of missing values'], inplace=True,ascending=False)
missing_values_df_test[missing_values_df_test['Percentage of missing values']>0][:10]
#BLOCKID can be dropped, since it is 43%missing values


# In[17]:


df_train.drop(columns=['BLOCKID', 'SUMLEVEL'],axis=1,inplace=True)
#SUMLEVEL does not have any predictive power and no variance


# In[18]:


df_test.drop(columns=['BLOCKID', 'SUMLEVEL'],axis=1,inplace=True)
#SUMLEVEL does not have any predictive power


# In[19]:


#imputing missing values with mean
missing_train_cols=[]
for col in df_train.columns:
    if df_train[col].isna().sum() !=0:
        missing_train_cols.append(col)
print(missing_train_cols)


# In[20]:


#imputing missing values with mean
missing_test_cols=[]
for col in df_test.columns:
    if df_test[col].isna().sum() !=0:
        missing_test_cols.append(col)
print(missing_test_cols)


# In[21]:


#Missing cols are all numerical variables
for col in df_train.columns:
    if col in (missing_train_cols):
        df_train[col].replace(np.nan,df_train[col].mean(),inplace=True)


# In[22]:


#Missing cols are all numerical variables
for col in df_test.columns:
    if col in (missing_test_cols):
        df_test[col].replace(np.nan,df_test[col].mean(),inplace=True)


# In[23]:


df_train.isna().sum().sum()


# In[24]:


df_test.isna().sum().sum()


# # Exploratory Data Analysis (EDA):

# ### 4.Perform debt analysis. You may take the following steps:

# ###### a) Explore the top 2,500 locations where the percentage of households with a second mortgage is the highest and percent ownership is above 10 percent. Visualize using geo-map. You may keep the upper limit for the percent of households with a second mortgage to 50 percent

# In[25]:


import plotly.express as px
import plotly.graph_objects as go


# In[26]:


df_train_location_mort_pct_s = df_train_location_mort_pct['lat'].astype(str).str.cat(sep=', ')


# In[ ]:


fig = go.Figure(data=go.Scattergeo(
    lat = df_train['lat'],
    lon = df_train['lng']),
    )
fig.update_layout(
    geo=dict(
        scope = 'north america',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
        projection = dict(
            type = 'conic conformal',
            rotation_lon = -100
        ),
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ -140.0, -55.0 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [ 20.0, 60.0 ],
            dtick = 5
        )
    ),
    title='Top 2,500 locations with second mortgage is the highest and percent ownership is above 10 percent')
fig.show()


# ###### b) Use the following bad debt equation: Bad Debt = P (Second Mortgage ∩ Home Equity Loan) Bad Debt = second_mortgage + home_equity - home_equity_second_mortgage c) Create pie charts to show overall debt and bad debt

# In[ ]:


df_train['bad_debt']=df_train['second_mortgage']+df_train['home_equity']-df_train['home_equity_second_mortgage']


# In[ ]:


df_train['bins'] = pd.cut(df_train['bad_debt'],bins=[0,0.10,1], labels=["less than 50%","50-100%"])
df_train.groupby(['bins']).size().plot(kind='pie',subplots=True,startangle=90, autopct='%1.1f%%')
plt.axis('equal')
plt.show()


# ###### d) Create Box and whisker plot and analyze the distribution for 2nd mortgage, home equity, good debt, and bad debt for different cities

# In[ ]:


cols=[]
df_train.columns


# In[ ]:


#Taking Las Vegas and Los Angeles cities Data
cols=['second_mortgage','home_equity','debt','bad_debt']
df_box_LV=df_train.loc[df_train['city'] == 'Las Vegas']
df_box_LA=df_train.loc[df_train['city'] == 'Los Angeles']
df_box_city=pd.concat([df_box_LV,df_box_LA])
df_box_city.head(4)


# In[ ]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='second_mortgage',y='city',width=0.5,palette="Set3")
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='home_equity',y='city',width=0.5,palette='Set3')
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='debt',y='city',width=0.5,palette='Set3')
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='bad_debt',y='city',width=0.5,palette='Set3')
plt.show()


# ###### e) Create a collated income distribution chart for family income, house hold income, and remaining income

# In[ ]:


sns.distplot(df_train['hi_mean'])
plt.title('Household income distribution chart')
plt.show()


# In[ ]:


sns.distplot(df_train['family_mean'])
plt.title('Family income distribution chart')
plt.show()


# In[ ]:


sns.distplot(df_train['family_mean']-df_train['hi_mean'])
plt.title('Remaining income distribution chart')
plt.show()


# # Exploratory Data Analysis (EDA):

# ###### 1. Perform EDA and come out with insights into population density and age. You may have to derive new fields (make sure to weight averages for accurate measurements):

# In[27]:


#plt.figure(figsize=(25,10))
fig,(ax1,ax2,ax3)=plt.subplots(3,1)
sns.distplot(df_train['pop'],ax=ax1)
sns.distplot(df_train['male_pop'],ax=ax2)
sns.distplot(df_train['female_pop'],ax=ax3)
plt.subplots_adjust(wspace=0.8,hspace=0.8)
plt.tight_layout()
plt.show()


# In[28]:


#plt.figure(figsize=(25,10))
fig,(ax1,ax2)=plt.subplots(2,1)
sns.distplot(df_train['male_age_mean'],ax=ax1)
sns.distplot(df_train['female_age_mean'],ax=ax2)
plt.subplots_adjust(wspace=0.8,hspace=0.8)
plt.tight_layout()
plt.show()


# ###### a) Use pop and ALand variables to create a new field called population density

# In[29]:


df_train['population_density']=df_train['pop']/df_train['ALand']
df_test['population_density']=df_test['pop']/df_test['ALand']


# In[30]:


sns.distplot(df_train['population_density'])
plt.title('Population Density')
plt.show()


# In[31]:


sns.distplot(df_test['population_density'])
plt.title('Population Density')
plt.show()


# ###### b) Use male_age_median, female_age_median, male_pop, and female_pop to create a new field called median age c) Visualize the findings using appropriate chart type

# In[32]:


df_train['age_median']=(df_train['male_age_median']+df_train['female_age_median'])/2
df_test['age_median']=(df_test['male_age_median']+df_test['female_age_median'])/2


# In[33]:


df_train[['male_age_median','female_age_median','male_pop','female_pop','age_median']].head()


# In[34]:


sns.distplot(df_train['age_median'])
plt.title('Median Age')
plt.show()


# In[35]:


sns.boxplot(df_train['age_median'])
plt.title('Population Density')
plt.show() 


# ### 2. Create bins for population into a new variable by selecting appropriate class interval so that the number of categories don’t exceed 5 for the ease of analysis.

# In[36]:


df_train['pop'].describe()


# In[37]:


df_train['pop_bins']=pd.cut(df_train['pop'],bins=5,labels=['very low','low','medium','high','very high'])


# In[38]:


df_train[['pop','pop_bins']]


# In[39]:


df_train['pop_bins'].value_counts()


# ###### a) Analyze the married, separated, and divorced population for these population brackets

# In[40]:


df_train.groupby(by='pop_bins')[['married','separated','divorced']].count()


# In[41]:


df_train.groupby(by='pop_bins')[['married','separated','divorced']].agg(["mean", "median"])


# ###### b) Visualize using appropriate chart type

# In[42]:


plt.figure(figsize=(10,5))
pop_bin_married=df_train.groupby(by='pop_bins')[['married','separated','divorced']].agg(["mean"])
pop_bin_married.plot(figsize=(20,8))
plt.legend(loc='best')
plt.show()


# ###### 3. Please detail your observations for rent as a percentage of income at an overall level, and for different states.

# In[43]:


rent_state_mean=df_train.groupby(by='state')['rent_mean'].agg(['mean'])
rent_state_mean.head()


# In[44]:


income_state_mean=df_train.groupby(by='state')['family_mean'].agg(["mean"])
income_state_mean.head()


# In[45]:


rent_perc_of_income=rent_state_mean['mean']/income_state_mean['mean']
rent_perc_of_income.head(10)


# In[46]:


#overall level rent as a percentage of income
sum(df_train['rent_mean'])/sum(df_train['family_mean'])


# ###### 4. Perform correlation analysis for all the relevant variables by creating a heatmap. Describe your findings.

# In[47]:


df_train.columns


# In[48]:


cor=df_train[['COUNTYID','STATEID','zip_code','type','pop', 'family_mean',
         'second_mortgage', 'home_equity', 'debt','hs_degree',
           'age_median','pct_own', 'married','separated', 'divorced']].corr()


# In[49]:


plt.figure(figsize=(20,10))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()


# ### Data Modeling : Linear Regression

# ###### 1.Build a linear Regression model to predict the total monthly expenditure for home mortgages loan. Please refer ‘deplotment_RE.xlsx’. Column hc_mortgage_mean is predicted variable. This is the mean monthly mortgage and owner costs of specified geographical location. Note: Exclude loans from prediction model which have NaN (Not a Number) values for hc_mortgage_mean.

# In[50]:


df_train.columns


# In[51]:


df_train['type'].unique()
type_dict={'type':{'City':1, 
                   'Urban':2, 
                   'Town':3, 
                   'CDP':4, 
                   'Village':5, 
                   'Borough':6}
          }
df_train.replace(type_dict,inplace=True)


# In[52]:


df_train['type'].unique()


# In[53]:


df_test.replace(type_dict,inplace=True)


# In[54]:


df_test['type'].unique()


# In[55]:


feature_cols=['COUNTYID','STATEID','zip_code','type','pop', 'family_mean',
         'second_mortgage', 'home_equity', 'debt','hs_degree',
           'age_median','pct_own', 'married','separated', 'divorced']


# In[56]:


x_train=df_train[feature_cols]
y_train=df_train['hc_mortgage_mean']


# In[57]:


x_test=df_train[feature_cols]
y_test=df_train['hc_mortgage_mean']


# In[58]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error,accuracy_score


# In[59]:


sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)


# #### a) Run a model at a Nation level. If the accuracy levels and R square are not satisfactory proceed to below step.

# In[60]:


linereg=LinearRegression()
linereg.fit(x_train_scaled,y_train)


# In[61]:


y_pred=linereg.predict(x_test_scaled)


# In[62]:


print("Overall R2 score of linear regression model", r2_score(y_test,y_pred))
print("Overall RMSE of linear regression model", np.sqrt(mean_squared_error(y_test,y_pred)))


# ###### b) Run another model at State level. There are 52 states in USA.

# In[63]:


state= df_train['STATEID'].unique()
state[0:5]


# In[64]:


for i in [20,1,45]:
    print("State ID-",i)
    
    x_train_nation=df_train[df_train['COUNTYID']==i][feature_cols]
    y_train_nation=df_train[df_train['COUNTYID']==i]['hc_mortgage_mean']
    
    x_test_nation=df_test[df_test['COUNTYID']==i][feature_cols]
    y_test_nation=df_test[df_test['COUNTYID']==i]['hc_mortgage_mean']
    
    x_train_scaled_nation=sc.fit_transform(x_train_nation)
    x_test_scaled_nation=sc.fit_transform(x_test_nation)
    
    linereg.fit(x_train_scaled_nation,y_train_nation)
    y_pred_nation=linereg.predict(x_test_scaled_nation)
    
    print("Overall R2 score of linear regression model for state,",i,":-" ,r2_score(y_test_nation,y_pred_nation))
    print("Overall RMSE of linear regression model for state,",i,":-" ,np.sqrt(mean_squared_error(y_test_nation,y_pred_nation)))
    print("\n")


# In[65]:


residuals=y_test-y_pred
residuals


# In[66]:


plt.hist(residuals)


# In[67]:


sns.distplot(residuals)


# In[68]:


plt.scatter(residuals,y_pred) 

