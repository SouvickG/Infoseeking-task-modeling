from __future__ import print_function
from __future__ import division
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import nltk.metrics.agreement
from string import printable


matt_annotations = pd.read_excel('/Users/Matt/Desktop/trec2014/2017-08-06/Matt - Task Facets_session track2014.xlsx',sheetname='Sheet1')
jiqun_annotations = pd.read_excel('/Users/Matt/Desktop/trec2014/2017-07-24/Task Facets_session track2014.xlsx',sheetname='Sheet1')
jiqun2_annotations = pd.read_excel('/Users/Matt/Desktop/trec2014/2017-09-23/Jiqun0923_Task Facets_session track2014_cognitive complexity.xlsx',sheetname='search task annotation')

# matt_product_annotations = matt_annotations['Task Product'].tolist()
# matt_product_annotations = [''.join(char for char in ano if char in printable) for ano in matt_product_annotations]
# matt_goal_annotations = matt_annotations['Task Goal'].tolist()
# matt_goal_annotations = [''.join(char for char in ano if char in printable) for ano in matt_goal_annotations]
#
# jiqun_product_annotations = jiqun_annotations['Task Product'].tolist()
# jiqun_product_annotations = [''.join(char for char in ano if char in printable) for ano in jiqun_product_annotations]
# jiqun_goal_annotations = jiqun_annotations['Task Goal'].tolist()
# jiqun_goal_annotations = [''.join(char for char in ano if char in printable) for ano in jiqun_goal_annotations]
# print(cohen_kappa_score(matt_product_annotations,jiqun_product_annotations))

print(cohen_kappa_score(jiqun_annotations['Task Product'],jiqun2_annotations['Task Product']))
print(cohen_kappa_score(jiqun_annotations['Task Goal'],jiqun2_annotations['Task Goal']))
exit()
print(cohen_kappa_score(matt_annotations['Task Product'],jiqun_annotations['Task Product']))
# print(cohen_kappa_score(matt_annotations['Task Goal'].tolist(),jiqun_annotations['Task Goal'].tolist()))
print(cohen_kappa_score(matt_annotations['Task Goal'],jiqun_annotations['Task Goal']))
print([x==y for (x,y) in zip(matt_annotations['Task Product'].tolist(),jiqun_annotations['Task Product'].tolist())])
# test_annotations = jiqun_annotations['Task Product'].tolist()
# test_annotations[-1]='wtf'
# print(matt_annotations['Task Goal'].tolist())
# print(jiqun_annotations['Task Goal'].tolist())
