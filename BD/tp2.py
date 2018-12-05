# -*- coding: utf-8 -*-

#instalar pip install apyori

import pandas as pd
import numpy as np
import csv


def main():
        
    df = pd.read_csv("drug_consumption.data", sep=",", index_col = 0)
    transactions = [list(x) for x in df.values]
    transactions = [[str(t) for t in list_trans] for list_trans in transactions]
    results = list(apriori(transactions, min_support=0.1, min_confidence = 0.6))    
    # min_lift, max_length
    results_df = pd.DataFrame(columns = ['Itemset','Support','A', 'B', "Confidence", "Lift"])
    
    for result in results:
        r = {"Itemset": list(result.items),\
         "Support": result.support,\
         "A": list(result.ordered_statistics[0][0]),\
         "B": list(result.ordered_statistics[0][1]),\
         "Confidence": result.ordered_statistics[0][2],  \
         "Lift": result.ordered_statistics[0][3]\
         }
        results_df = results_df.append(r, ignore_index=True)
    
    results_df.to_csv("resultados.csv", sep =";")


if __name__ == "__main__":
    main()
