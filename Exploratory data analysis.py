#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install sweetviz


# In[4]:


pip install pandas-profiling


# In[8]:


pip install seaborn


# In[52]:


import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


data = pd.read_csv(r'C:\Users\chait\Downloads\sample_dataset.csv')


# In[19]:


data.shape


# In[38]:


(data.isna().sum()/data.shape[0]).sort_values().plot(kind='bar')


# In[44]:


(data.describe(percentiles=[0.05,0.25,0.5,0.75,0.95]))


# In[53]:


data.hist(figsize=(100,100))
plt.show()


# In[49]:


data.iloc[:,0:4].hist()


# In[56]:


data['area error'].value_counts().plot(kind = 'bar')


# Box Plot
# 

# In[57]:


data.columns


# In[59]:


data[['mean radius']].plot(kind= 'box')


# In[60]:


data['mean radius'].hist()


# In[63]:


data[['mean radius','mean texture']].boxplot()


# In[65]:


data.iloc[:,0:5].boxplot()


# In[66]:


import seaborn as sns


# In[67]:


sns.pairplot(data.iloc[:,0:5])


# In[68]:


sns.pairplot(data.iloc[0:100,0:5],kind='hist')


# In[69]:


sns.pairplot(data.iloc[0:100,0:5],kind='kde')#kernal density estimation


# In[71]:


sns.pairplot(data.iloc[0:100,0:5],corner='True')


# In[72]:


sns.pairplot(data, x_vars=['mean radius','mean texture'], y_vars=['mean radius','mean texture','mean area'])


# In[73]:


sns.pairplot(data, x_vars=['mean radius','mean texture'], y_vars=['mean radius','mean texture','mean area'],hue='target')


# #correlation matrix

# In[74]:


data.corr()


# In[75]:


data.corr(method='spearman')


# #Heat map

# In[76]:


plt.figure(figsize=(12,8))
sns.heatmap(data.corr())
plt.show()


# In[77]:


plt.figure(figsize=(12,8))
sns.heatmap(data.corr().abs())
plt.show()


# # stacked histogram

# In[78]:


sns.histplot(data,x='mean radius',hue='target',multiple='stack')


# In[79]:


sns.histplot(data,x='worst concavity',hue='target',multiple='stack')


# In[80]:


sns.histplot(data,x='worst perimeter',hue='target',multiple='stack')


# In[88]:


columns = data.select_dtypes(exclude='object').drop('target',axis = 1).columns


# In[89]:


columns


# In[91]:


fig,axs = plt.subplots(len(columns),1,figsize=(5,5*len(columns)))
for i in range(len(columns)):
    column_name = columns[i]
    sns.histplot(data,x=column_name,hue='target',multiple='stack',ax = axs[i])


# # sweet viz

# In[92]:


import sweetviz


# In[93]:


report = sweetviz.analyze(data,target_feat='target')


# In[94]:


report.show_html(layout='vertical')


# In[95]:


report.show_notebook()


# In[96]:


report2 = sweetviz.compare([data.head(50),"dataset1"],[data.tail(50),"dataset2"],target_feat='target')


# In[98]:


report2.show_html(layout='vertical')


# # pandas profiling

# In[99]:


from pandas_profiling import ProfileReport


# In[101]:


report = ProfileReport(data.iloc[:,0:5])


# In[102]:


report.to_notebook_iframe()


# In[105]:


report.to_file("report.html",silent=False)


# # avoid the calculation of multivariate statistics

# In[106]:


report = ProfileReport(data.iloc[:,0:5],minimal=True)
report.to_notebook_iframe()


# In[ ]:




