import pandas as pd


data=pd.read_csv('Data/datasetSentences.txt',delimiter='\t')
label=pd.read_csv('Data/datasetSplit.txt')


final=data.merge(label,on='sentence_index')

final=final[['sentence','splitset_label']]
final.columns=['sentence','label']

train=final[:round(len(final)/2)]
test=final[round(len(final)/2):]


train.to_csv('Data/train.csv',index=None)
test.to_csv('Data/test.csv',index=None)