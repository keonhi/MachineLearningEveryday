#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import jarque_bera
from scipy.stats import norm


# In[2]:


data=pd.read_excel('index_rates_data.xlsx')
data=data.rename(columns={'Unnamed: 0': 'Date'})
data.index=data['Date']
data=data.drop(columns=['Date'])
data=data.sort_index()


# In[3]:


data


# In[4]:


data['SPX'].pct_change().std()*np.sqrt(12)


# In[5]:


data['10Y interest rate'].pct_change().std()*np.sqrt(12)


# In[6]:


price_history=data.copy()


# In[7]:


price_history['10YT']=100/(1+price_history['10Y interest rate']/100)**10
price_history=price_history.drop(columns=['10Y interest rate'])


# In[8]:


price_history


# In[9]:


price_history['10YT'].pct_change().std()*np.sqrt(12)


# In[10]:


initial_rate=data['10Y interest rate'].iloc[-1]/100
initial_SPX=data['SPX'].iloc[-1]


# In[11]:


np.random.seed(0)


# In[12]:


random_series_volatility=pd.DataFrame(columns=['Interest_Rate', '10YT'])


# In[13]:


random_series=pd.DataFrame(index=price_history.index, columns=['Interest_Rate', 'Rate_Change', '10YT', 'SPX', '10YT_Return', 'SPX_Return', '10YT_Return_Short'])
random_series['Interest_Rate'].iloc[0]=initial_rate
random_series['SPX'].iloc[0]=initial_SPX


# In[14]:


u=np.random.normal(0, 1, random_series.shape[0]-1)
v=np.random.normal(0, 1, random_series.shape[0]-1)


# In[15]:


random_series['Rate_Change'].iloc[1:]=0.40/np.sqrt(12)*u
random_series['SPX_Return'].iloc[1:]=0.20/np.sqrt(12)*(-0.40*u+np.sqrt(1-(-0.40)**2)*v)


# In[16]:


for i in range(1, random_series.shape[0]):
    random_series['Interest_Rate'].iloc[i]=    random_series['Interest_Rate'].iloc[i-1]*(1+random_series['Rate_Change'].iloc[i])


# In[17]:


for i in range(1, random_series.shape[0]):
    random_series['SPX'].iloc[i]=    random_series['SPX'].iloc[i-1]*(1+random_series['SPX_Return'].iloc[i])


# In[18]:


random_series['10YT']=100/(1+random_series['Interest_Rate'])**10


# In[19]:


random_series['10YT_Return']=random_series['10YT'].pct_change()


# In[20]:


random_series['10YT_Return_Short']=-random_series['10YT'].pct_change()


# In[21]:


random_series


# In[22]:


random_series['SPX_Return'].std()*np.sqrt(12)


# In[23]:


random_series['Rate_Change'].std()*np.sqrt(12)


# In[24]:


random_series['10YT'].pct_change().std()*np.sqrt(12)


# In[25]:


random_series[['Rate_Change','SPX_Return']].astype(float).corr().iloc[0,1]


# In[26]:


random_series[['10YT_Return','SPX_Return']].astype(float).corr().iloc[0,1]


# In[27]:


random_series[['10YT_Return_Short','SPX_Return']].astype(float).corr().iloc[0,1]


# In[28]:


num_trials=10000
random_series_variance=pd.DataFrame(columns=['Interest_Rate', '10YT', 'SPX', '10YT_Short', 'corr_1', 'corr_2', 'corr_3'])


# In[29]:


get_ipython().run_cell_magic('time', '', "for trial in range(num_trials):\n    random_series=pd.DataFrame(index=price_history.index, columns=['Interest_Rate', 'Rate_Change', '10YT', 'SPX', '10YT_Return', 'SPX_Return', '10YT_Return_Short'])\n    random_series['Interest_Rate'].iloc[0]=initial_rate\n    random_series['SPX'].iloc[0]=initial_SPX\n\n    u=np.random.normal(0, 1, random_series.shape[0]-1)\n    v=np.random.normal(0, 1, random_series.shape[0]-1)\n\n    random_series['Rate_Change'].iloc[1:]=0.40/np.sqrt(12)*u\n    random_series['SPX_Return'].iloc[1:]=0.20/np.sqrt(12)*(-0.40*u+np.sqrt(1-(-0.40)**2)*v)\n\n    for i in range(1, random_series.shape[0]):\n        random_series['Interest_Rate'].iloc[i]=\\\n        random_series['Interest_Rate'].iloc[i-1]*(1+random_series['Rate_Change'].iloc[i])\n\n    for i in range(1, random_series.shape[0]):\n        random_series['SPX'].iloc[i]=\\\n        random_series['SPX'].iloc[i-1]*(1+random_series['SPX_Return'].iloc[i])\n\n    random_series['10YT']=100/(1+random_series['Interest_Rate'])**10\n\n    random_series['10YT_Return']=random_series['10YT'].pct_change()\n    random_series['10YT_Return_Short']=-random_series['10YT'].pct_change()\n    \n    random_series_variance.loc[trial, ['Interest_Rate', '10YT', 'SPX']]=\\\n    (random_series[['Interest_Rate', '10YT', 'SPX']].pct_change().std()*np.sqrt(12))**2\n    \n    random_series_variance.loc[trial]['10YT_Short']= (random_series['10YT_Return_Short'].std()*np.sqrt(12))**2\n    \n    random_series_variance.loc[trial]['corr_1']=random_series[['Rate_Change','SPX_Return']].astype(float).corr().iloc[0,1]\n    random_series_variance.loc[trial]['corr_2']=random_series[['10YT_Return','SPX_Return']].astype(float).corr().iloc[0,1]\n    random_series_variance.loc[trial]['corr_3']=random_series[['10YT_Return_Short','SPX_Return']].astype(float).corr().iloc[0,1]")


# In[30]:


random_series_variance


# In[31]:


( (random_series_variance['Interest_Rate']*(random_series.shape[0]-1)).sum()/((random_series.shape[0]-1)*num_trials-num_trials))**0.5


# In[32]:


( (random_series_variance['10YT']*(random_series.shape[0]-1)).sum()/((random_series.shape[0]-1)*num_trials-num_trials))**0.5


# In[42]:


( (random_series_variance['10YT_Short']*(random_series.shape[0]-1)).sum()/((random_series.shape[0]-1)*num_trials-num_trials))**0.5


# In[33]:


( (random_series_variance['SPX']*(random_series.shape[0]-1)).sum()/((random_series.shape[0]-1)*num_trials-num_trials))**0.5


# In[34]:


random_series_variance['corr_1'].mean()


# In[35]:


random_series_variance['corr_2'].mean()


# In[36]:


random_series_variance['corr_3'].mean()


# In[43]:


w1=0.31
w2=0.69
s1=0.20
s2=0.09
rho=-0.35


# In[44]:


A=-4000000/np.sqrt(w1**2*s1**2+w2**2*s2**2+2*w1*w2*rho*s1*s2)/norm.ppf(1/12)


# In[45]:


A


# In[46]:


A*w1


# In[47]:


A*w2

