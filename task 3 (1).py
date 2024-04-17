#!/usr/bin/env python
# coding: utf-8

# # Lets load the boston house pricing datasets 

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[89]:


boston=pd.read_csv(r"C:\Users\muska\Downloads\boston csv.csv")


# In[90]:


boston


# In[84]:


import sklearn
print(sklearn.__version__)


# In[86]:


get_ipython().system('pip install scikit-learn==1.1.3')


# In[ ]:





# In[91]:


boston.keys()


# In[31]:


## lets check the description of the dataset
print(df.AGE)


# In[94]:


print(df.INDUS)


# In[ ]:





# In[ ]:





# ## preparing the dataset

# In[48]:


dataset=pd.DataFrame(boston.data)


# In[49]:


print(dataset)


# In[50]:


dataset.head()


# In[51]:


dataset['price']=boston.target


# In[52]:


dataset.head()


# In[53]:


dataset.info()


# In[54]:


## summarzing The Stats of the data
dataset.describe()


# In[58]:


## Check the missing values 
dataset.isnull()


# In[59]:


dataset.isnull().sum()


# In[61]:


### Exploratory Data Analysis
## Correlation
dataset.corr()


# In[62]:


import seaborn as sns 
sns.pairplot(dataset)


# In[67]:


plt.scatter(dataset['CRIM'],dataset['price'])
plt.xlabel("Crime Rate")
plt.ylabel("price")


# In[68]:


plt.scatter(dataset['RM'],dataset['price'])
plt.xlabel("RM")
plt.ylabel("price")


# In[69]:


import seaborn as sns 
sns.regplot(x="RM",y="price",data=dataset)


# In[70]:


sns.regplot(x="LSTAT",y="price",data=dataset)


# In[79]:


sns.regplot(x="PTRATIO",y="price",data=dataset)


# In[81]:


## Independent and Dependent features

X=df.iloc[:,:-1]
y=df.iloc[:,:-1]


# In[82]:


X


# In[80]:


y


# In[83]:


## Train Test Split 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[52]:


X_train


# In[53]:


X_test


# In[54]:


y_train


# In[55]:


y_test


# In[56]:


import sklearn
print(sklearn.__version__)


# In[77]:


get_ipython().system('pip install scikit-learn==1.1.3')


# In[57]:


## standardize the dataset
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()


# In[97]:


import sklearn
from sklearn.datasets import load_boston


# In[58]:


X_train=Scaler.fit_transform(X_train)


# In[59]:


X_tset=Scaler.transform(X_train)


# In[60]:


X_train


# In[61]:


X_test


#  ## Model Training

# In[62]:


from sklearn.linear_model import LinearRegression


# In[63]:


regression=LinearRegression()


# In[64]:


regression.fit(X_train,y_train)


# In[65]:


## print the cofficients and the intercept
print(regression.coef_)


# In[66]:


print(regression.intercept_)


# In[67]:


## on which parameters the model has been trained 
regression.get_params()


# In[68]:


### Prediction with test data 
reg_pred=regression.predict(X_train)


# In[69]:


reg_pred


# In[98]:


## plot a scatter plot for the prediction
plt.scatter(X_train,reg_pred)


# In[ ]:





# In[71]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_squared_error(y_train,reg_pred))
print(mean_squared_error(y_train,reg_pred))


# ## R square and adjusted R square

# In[7]:


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)


# In[ ]:


#display adjusted R-squared
1 - (1-score)*(len(y-test)-1)/(len(y_test)-X_test.shape[1]-1)


# ## New Data Prediction

# In[112]:


boston.data[0].reshap(1,-1)


# In[114]:


## transformation of new data
scaler.transform(boston.data[0].reshape(1,-1))


# In[115]:


regression.predict(scalar.transform(boston.data[0].reshape(1,-1)))


# ## pickling the model file for deploment

# In[ ]:


import pickle


# In[ ]:


pickle.dumb(regression,open('regmodel.pkl'),'wb')


# In[116]:


pickled_nodel=pickle.load((opem("regmodel.pkl',")'regmbol.rb')


# In[117]:


## prediction
picled_model.prdict(scalar.transform(boston.data[0].reshape(1,-1))


# To expand on the learning outcomes of the Boston CSV framework and reach approximately 6000 words, we can delve into each outcome in detail, providing examples, strategies, and case studies to illustrate their significance and implementation. Here's a structured outline to guide the expansion:
# 
# ---
# 
# *Title: Understanding the Learning Outcomes of the Boston CSV Framework*
# 
# *Introduction:*
# - Brief overview of the Boston CSV framework
# - Importance of customer success in modern business
# - Purpose of outlining learning outcomes
# 
# *1. Understanding Customer Needs:*
# - Importance of understanding customer needs
# - Techniques for gathering customer insights (surveys, interviews, data analysis)
# - Case studies of companies excelling in customer understanding (e.g., Amazon, Netflix)
# - Strategies for leveraging customer feedback to improve products/services
# 
# *2. Aligning Products/Services with Customer Needs:*
# - Importance of aligning offerings with customer needs
# - Methods for product/service alignment (market research, product iteration)
# - Case studies of companies adapting offerings to meet customer demands (e.g., Apple, Airbnb)
# - Strategies for continuous improvement and innovation in product/service alignment
# 
# *3. Building Strong Customer Relationships:*
# - Significance of building strong relationships with customers
# - Approaches to relationship-building (personalization, communication channels)
# - Case studies of companies with exemplary customer relationship management (e.g., Zappos, Salesforce)
# - Strategies for nurturing long-term customer relationships and loyalty
# 
# *4. Measuring and Improving Customer Satisfaction:*
# - Importance of measuring customer satisfaction
# - Key metrics for evaluating customer satisfaction (Net Promoter Score, Customer Satisfaction Score)
# - Case studies of companies with effective satisfaction measurement and improvement strategies (e.g., Starbucks, Disney)
# - Strategies for continuous feedback collection and analysis to drive satisfaction improvements
# 
# *5. Driving Long-Term Customer Success and Loyalty:*
# - Significance of prioritizing long-term customer success and loyalty
# - Methods for fostering customer success (customer education, proactive support)
# - Case studies of companies achieving long-term customer success and loyalty (e.g., Adobe, HubSpot)
# - Strategies for maintaining customer engagement and advocacy over time
# 
# *Conclusion:*
# - Recap of the importance of the Boston CSV framework
# - Summary of key learning outcomes and their significance
# - Call to action for businesses to prioritize customer success
# 
# *References:*
# - Cite relevant research papers, articles, and resources supporting the discussed topics
# 
# ---
# 
# By elaborating on each learning outcome with detailed explanations, examples, and strategies, we can reach the desired word count while providing valuable insights into the Boston CSV framework and its application in business contexts.

# In[ ]:





# In[ ]:




