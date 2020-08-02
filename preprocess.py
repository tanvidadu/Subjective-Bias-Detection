import pandas as pd
import sys
data_split=sys.argv[1]
data = pd.read_csv('data/'+data_split+'.tsv', sep='\t')
data.head()

df_bert = pd.DataFrame({
        'id':range(len(data)),
        'alpha':['a']*data.shape[0],
        'label':data['label'],
        'text':data['Text'].replace(r'\n', ' ', regex=True)
    })

df_bert.head()
df_bert.to_csv('data/final'+data_split+'.tsv', sep='\t', index=None, header=None)
