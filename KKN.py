#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[6]:


#Cargamos nuestro dataset
pd.read_csv('Raw-Data.csv')
df = pd.read_csv('Raw-Data.csv')


# In[8]:


# Eliminanos las columnas que no utilizaremos
df = df.drop(['Severity'], axis=1)


# In[11]:


# Rempleazamos los valores no numericos con la mediana
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())


# In[12]:


# Convertir las cadena de caracteres en valores numericos
df = pd.get_dummies(df, columns =['Gender', 'Contact'])


# In[20]:


# Dividimos los datos en conjuntos de datos de prueba y de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(df.drop('Severity', axis=1), df['Severity'], test_size=0.4, random_state=3)


# In[ ]:




