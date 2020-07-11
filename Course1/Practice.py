# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:26:27 2019

@author: 496070
"""
import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(12).reshape(3,4),columns=['A', 'B', 'C', 'D'])
df['E']= 4
df.iloc(2)
df = df[df['A'] != df['E']]
df

df['E']

df['A']== df['E']
df = df.drop(df['A']== df['E'])
