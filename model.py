# -*- coding: utf-8 -*
import pandas as pd
#!pip install word2number
from word2number import w2n
from sklearn.linear_model import LinearRegression
import pickle

df=pd.read_csv('hiring.csv')
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean())
df['experience']=df['experience'].fillna('zero')
df['experience']=df['experience'].apply(w2n.word_to_num)
x=df.drop('salary($)',axis='columns')
y=df['salary($)']
model=LinearRegression()
model.fit(x,y)
print( model.predict([[2,9,6]]))
with open ('hiring.pickle','wb') as f:
    pickle.dump(model,f)
