# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
from scipy.stats.mstats import zscore


df= pd.read_csv("drug_consumption.data.csv", index_col = 0)

cols_subs = ["alcool", "anfetamina", "nitrato_amilato", "benzodiazepina", "cafeina", "maconha", "chocolate", "cocaina", "crack", "ecstase", "heroina", "ketamina", "legalidade", "lsd", "metadona", "cogumelos", "nicotina", "semeron", "vsa"]
cols_del = ["pais", "etnia"]

vals_dict = {"CL"+str(i): i for i in range(7)}

def substituir(data):
    return vals_dict[data]

for col in cols_subs:
    #df[col] = df[col].apply(substituir)
    df[col] = zscore(df[col])

df.drop(cols_del, axis=1, inplace=True)

df.to_csv("drogas_preprocessadas.csv")