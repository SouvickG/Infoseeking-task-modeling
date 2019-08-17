import pandas as pd
import mysql.connector
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

exp1_csv_folder = '/Users/Matt/Desktop/Experiment/Data/Interim/structure_graph_metrics/'


dataset_name_to_printable_exp1 = {
'nsf_full':'INT (full)',
'nsf_subset':'INT',
'sal_full':'SAL (full)',
'sal_subset':'SAL',
'eopn_full':'EOP (full)',
'eopn_subset':'EOP',
'nsf_eopn':'INT,EOPN',
'nsf_sal':'INT,SAL',
'sal_eopn':'EOPN,SAL',
'nsf_sal_eopn':'INT,EOPN,SAL',
}
algorithm_name_to_printable_exp1 = {
'mmhc':'MMHC',
'hc':'HC',
'manual':'Manual'
}

#TODO: Put manual
algorithms = ['mmhc','hc']

scores_exp1 = ['AIC','BIC','logLik']
#Experiment 1 table
df = []
for data_name in dataset_name_to_printable_exp1.keys():
    features = ''
    if '_full' in data_name:
        features = 'Full'
    else:
        features = 'Common'
# for data_name in ['eopn_full','eopn_subset','nsf_full','nsf_subset','sal_full','sal_subset','nsf_eopn']:
    data_name_printable = dataset_name_to_printable_exp1[data_name]
    dataset_df = pd.read_csv(exp1_csv_folder+'%s_metrics.csv'%(data_name))
    for algorithm in algorithms:
        subdf = dataset_df[dataset_df['algorithm']==algorithm]
        max_n = max(subdf['n'].tolist())
        subdf = subdf[subdf['n']==max_n]
        AIC = subdf['AIC'].mean().round(decimals=2)
        BIC = subdf['BIC'].mean().round(decimals=2)
        logLik = subdf['logLik'].mean().round(decimals=2)
        df += [{
            'Data':data_name_printable,
            'Algorithm':algorithm_name_to_printable_exp1[algorithm],
            'Features':features,
            'AIC':np.format_float_scientific(AIC,precision=2),
            'BIC':np.format_float_scientific(BIC,precision=2),
            'LL':np.format_float_scientific(logLik,precision=2)
        }]

for data_name in dataset_name_to_printable_exp1.keys():
    features = ''
    if '_full' in data_name:
        features = 'Full'
    else:
        features = 'Common'
# for data_name in ['eopn_full','eopn_subset','nsf_full','nsf_subset','sal_full','sal_subset','nsf_eopn']:
    data_name_printable = dataset_name_to_printable_exp1[data_name]
    dataset_df = pd.read_csv(exp1_csv_folder+'manual_%s_metrics.csv'%(data_name))
    subdf = dataset_df[dataset_df['algorithm']=='manual']
    max_n = max(subdf['n'].tolist())
    subdf = subdf[subdf['n']==max_n]
    AIC = subdf['AIC'].mean().round(decimals=2)
    BIC = subdf['BIC'].mean().round(decimals=2)
    logLik = subdf['logLik'].mean().round(decimals=2)
    df += [{
        'Data':data_name_printable,
        'Algorithm':'Manual',
        'Features':features,
        'AIC':np.format_float_scientific(AIC,precision=2),
        'BIC':np.format_float_scientific(BIC,precision=2),
        'LL':np.format_float_scientific(logLik,precision=2)
    }]
df = pd.DataFrame(df)
print df.sort_values(['Data','Features','Algorithm']).set_index(['Data','Features','Algorithm']).to_latex()


print("BAYES FACTOR")


df = []
n_total_edges = {
'eopn_full':45,
'eopn_subset':36,
'nsf_full':105,
'nsf_subset':36,
'sal_full':55,
'sal_subset':36,
'nsf_sal':36,
'nsf_eopn':36,
'sal_eopn':36,
'nsf_sal_eopn':36,
}

for data_name in ['eopn_full','eopn_subset','nsf_full','nsf_subset','sal_full','sal_subset']:
    features = ''
    if '_full' in data_name:
        features = 'Full'
    else:
        features = 'Common'
    data_name_printable = dataset_name_to_printable_exp1[data_name]
    dataset_df = pd.read_csv(exp1_csv_folder+'manualpairwise_datasetname1_%s_datasetname2_%s_metrics.csv'%(data_name,data_name))
    for algorithm in algorithms:
        subdf = dataset_df[dataset_df['algorithm1']==algorithm]
        BF = subdf['BF_log'].mean().round(decimals=2)
        SHD = subdf['SHD'].mean()
        scaled_SHD = SHD/float(n_total_edges[data_name])
        scaled_SHD = scaled_SHD.round(decimals=2)
        scaled_SHD = scaled_SHD.round(decimals=2)
        n_arcs = subdf['n_arcs1'].mean().round(decimals=2)
        prec = subdf['precision'].mean().round(decimals=2)
        rec = subdf['recall'].mean().round(decimals=2)
        df += [{
            'Data':data_name_printable,
            'Features':features,
            'Algorithm':algorithm_name_to_printable_exp1[algorithm],
            'log(BF)':BF,
            'Scaled SHD':scaled_SHD,
            '|E|':n_arcs,
            'Pr':prec,
            'Rec':rec,
        }]
print pd.DataFrame(df).sort_values(['Data','Features','Algorithm']).set_index(['Data','Features','Algorithm']).to_latex()
# df = []
# for i in range(1,6):
#     df += [{'dataset':'dataset%d'%i,'algo':1,'val':np.random.random()}]
#     df += [{'dataset':'dataset%d'%i,'algo':2,'val':np.random.random()}]
#     df += [{'dataset':'dataset%d'%i,'algo':3,'val':np.random.random()}]
#



##########
#
# Experiment 2
#
##########

dataset_name_to_printable_exp2 = {
'nsf_subset':'INT',
'sal_subset':'SAL',
'eopn_subset':'EOP',
'nsf_eopn':'INT,EOPN',
'nsf_sal':'INT,SAL',
'sal_eopn':'EOPN,SAL',
'nsf_sal_eopn':'INT,EOPN,SAL',
}

df = []



exp2_csv_folder = '/Users/Matt/Desktop/Experiment/Data/Interim/structure_graph_metrics/'
# write.csv(pairwise_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'manualpairwise_','datasetname1_',datasetname1,'_datasetname2_',datasetname2,'_metrics.csv',sep=''))
# write.csv(pairwise_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'fullpairwise_','datasetname1_',datasetname1,'_datasetname2_',datasetname2,'_metrics.csv',sep=''))
i = 0
for datasetname1 in dataset_name_to_printable_exp2.keys():
    j = 0
    for datasetname2 in dataset_name_to_printable_exp2.keys():
        if False:
        # if (datasetname1.startswith('eopn') or datasetname2.startswith('eopn')):
        # if ((datasetname1.startswith('eopn')) or (datasetname2.startswith('eopn'))) and not ((datasetname1.startswith('eopn')) and (datasetname2.startswith('eopn'))):
            value = '-'
        else:

            d = pd.read_csv(exp2_csv_folder+'fullpairwise_datasetname1_%s_datasetname2_%s_metrics.csv'%(datasetname1,datasetname2))
            shd = d['SHD']
            shd_mmhc = (d[d['algorithm1']=='mmhc']['SHD'].mean()/36.0).round(decimals=2)
            shd_hc = (d[d['algorithm1']=='hc']['SHD'].mean()/36.0).round(decimals=2)
            value=shd.mean().round(decimals=2)
            if i > j:
                value = shd_hc
            elif i < j:
                value = shd_mmhc
            elif i == j:
                value = str(shd_hc) + '/' + str(shd_mmhc)
            # if i > j:
            #     value = 'low'
            # elif i < j:
            #     value = 'up'
            # elif i == j:
            #     value ='low/up'
        df += [{
        'dataset1':'%d'%i+dataset_name_to_printable_exp2[datasetname1],
        'dataset2':'%d'%j+dataset_name_to_printable_exp2[datasetname2],
        'value':value
        # 'value':value/float(36)
        }]
        j+=1
    i+=1
df = pd.DataFrame(df).pivot(index='dataset1', columns='dataset2', values='value').to_latex()
print df



# df = []
# i = 0
# for datasetname1 in dataset_name_to_printable_exp2.keys():
#     j = 0
#     for datasetname2 in dataset_name_to_printable_exp2.keys():
#         if False:
#         # if (datasetname1.startswith('eopn') or datasetname2.startswith('eopn')):
#         # if ((datasetname1.startswith('eopn')) or (datasetname2.startswith('eopn'))) and not ((datasetname1.startswith('eopn')) and (datasetname2.startswith('eopn'))):
#             value = '-'
#         else:
#
#             d = pd.read_csv(exp2_csv_folder+'fullpairwise_datasetname1_%s_datasetname2_%s_metrics.csv'%(datasetname1,datasetname2))
#             bflogmax = d['BF_log_max']
#             bflogmax_mmhc = (d[d['algorithm1']=='mmhc']['BF_log_max'].mean()).round(decimals=2)
#             bflogmax_hc = (d[d['algorithm1']=='hc']['BF_log_max'].mean()).round(decimals=2)
#             value=shd.mean().round(decimals=2)
#             if i > j:
#                 value = bflogmax_hc
#             elif i < j:
#                 value = bflogmax_mmhc
#             elif i == j:
#                 value = str(bflogmax_hc) + '/' + str(bflogmax_mmhc)
#             # if i > j:
#             #     value = 'low'
#             # elif i < j:
#             #     value = 'up'
#             # elif i == j:
#             #     value ='low/up'
#         df += [{
#         'dataset1':'%d'%i+dataset_name_to_printable_exp2[datasetname1],
#         'dataset2':'%d'%j+dataset_name_to_printable_exp2[datasetname2],
#         'value':value
#         # 'value':value/float(36)
#         }]
#         j+=1
#     i+=1
# df = pd.DataFrame(df).pivot(index='dataset1', columns='dataset2', values='value').to_latex()
# print df

df = []
for data_name in dataset_name_to_printable_exp2.keys():
    features = ''
    if '_full' in data_name:
        features = 'Full'
    else:
        features = 'Common'
    data_name_printable = dataset_name_to_printable_exp2[data_name]
    dataset_df = pd.read_csv(exp1_csv_folder+'fullpairwise_datasetname1_%s_datasetname2_%s_metrics.csv'%(data_name,data_name))
    for algorithm in algorithms:
        subdf = dataset_df[dataset_df['algorithm1']==algorithm]
        BF = subdf['BF_log_max'].mean().round(decimals=2)
        SHD = subdf['SHD'].mean()
        scaled_SHD = SHD/float(n_total_edges[data_name])
        scaled_SHD = scaled_SHD.round(decimals=2)
        scaled_SHD = scaled_SHD.round(decimals=2)
        n_arcs = subdf['n_arcs1'].mean().round(decimals=2)
        prec = subdf['precision'].mean().round(decimals=2)
        rec = subdf['recall'].mean().round(decimals=2)

        df += [{
            'Data':data_name_printable,
            'Features':features,
            'Algorithm':algorithm_name_to_printable_exp1[algorithm],
            'max(log(BF))':BF,
            'Pr':prec,
            'Rec':rec,
        }]
print pd.DataFrame(df).sort_values(['Data','Features','Algorithm']).set_index(['Data','Features','Algorithm']).to_latex()






##########
#
# Prob edges total
#
##########

names = {
'time_dwelltotal_total_content_segment':'Time Content',
'facet_goal_val_amorphous':"Goal",
'facet_product_val_intellectual':"Product",
'topic_familiarity':"Topic Familiarity",
'task_difficulty':"Difficulty",
'query_length':'Query length',
'time_dwelltotal_total_serp_segment':'Time SERP',
'search_expertise':'Search Expertise',
'pages_num_segment':'# Pages',
'post_rushed':'Adequate Time',
'search_years':'Searcy Years',
'search_frequency':'Search Frequency',
'assignment_experience':'Task Experience',
'search_journalism':'Journalism Experience',
'topic_globalwarming':'Topic'
}
edges_df = pd.read_csv(exp1_csv_folder+'edges.csv')
edges_df = edges_df[~edges_df['datasetname'].isin(['nsf_full','sal_full','eopn_full'])]
edges_test_groupby = edges_df.groupby(['datasetname','downsample_number','randomsample_number','bootstrap_num','algorithm'])

n_total_tests = len(edges_test_groupby)
edges_total_occurrences = dict()
for ((edgefrom,to),group) in edges_df.groupby(['from','to']):
    edges_total_occurrences[(edgefrom,to)] = len(group.index)


df = []
for (edgefrom,to) in edges_total_occurrences.keys():
    df += [{
        'from':names[edgefrom],
        'to':names[to],
        'Pr':np.round(edges_total_occurrences[(edgefrom,to)]/float(n_total_tests),2),
    }]
df = pd.DataFrame(df)
print df.sort_values(['Pr','from','to'],ascending=[False,True,True])[['from','to','Pr']].head(15).to_latex()
##########
#
# Prob edges per data set
#
##########

dataset_name_to_printable_exp2 = {
'nsf_full':'INT (full)',
'sal_full':'SAL (full)',
'eopn_full':'EOP (full)',
'nsf_subset':'INT',
'sal_subset':'SAL',
'eopn_subset':'EOP',
'nsf_eopn':'INT,EOPN',
'nsf_sal':'INT,SAL',
'sal_eopn':'EOPN,SAL',
'nsf_sal_eopn':'INT,EOPN,SAL',
}
edges_df_sub = pd.read_csv(exp1_csv_folder+'edges.csv')
edges_df_sub = edges_df_sub[edges_df_sub['datasetname'].isin(['nsf_full','sal_full','eopn_full'])]

df = []
for (datasetname,edges_df) in edges_df_sub.groupby('datasetname'):
    edges_test_groupby = edges_df.groupby(['downsample_number','randomsample_number','bootstrap_num','algorithm'])
    n_total_tests = len(edges_test_groupby)
    edges_total_occurrences = dict()
    for ((edgefrom,to),group) in edges_df.groupby(['from','to']):
        edges_total_occurrences[(edgefrom,to)] = len(group.index)



    df_dataset = []
    for (edgefrom,to) in edges_total_occurrences.keys():
        df_dataset += [{
            'datasetname':dataset_name_to_printable_exp2[datasetname],
            'from':names[edgefrom],
            'to':names[to],
            'Pr':np.round(edges_total_occurrences[(edgefrom,to)]/float(n_total_tests),2),
        }]
    df_dataset = pd.DataFrame(df_dataset)
    df_dataset = df_dataset.sort_values(['Pr','datasetname','from','to'],ascending=[False,True,True,True]).head(5)
    df += [df_dataset]
df = pd.concat(df)
print df.sort_values(['datasetname','Pr','from','to'],ascending=[True,False,True,True])[['datasetname','from','to','Pr']].to_latex()

##########
#
# Experiment 3
#
##########

exp3_scores_to_colnames = {
'AIC':'AIC',
'BIC':'BIC',
'logLik':'LL',
}

dataset_name_to_printable_exp3 = {
'nsf_full':'INT (full)',
'nsf_subset':'INT',
'sal_full':'SAL (full)',
'sal_subset':'SAL',
'eopn_full':'EOP (full)',
'eopn_subset':'EOP',
'nsf_eopn':'INT,EOP',
'nsf_sal':'INT,SAL',
'sal_eopn':'EOP,SAL',
'nsf_sal_eopn':'INT,EOP,SAL',
}
for algorithm in algorithms:
    for score in exp3_scores_to_colnames.keys():
        dfs = []
        for dataset_name in dataset_name_to_printable_exp3.keys():
            dataset_df = pd.read_csv(exp1_csv_folder+'%s_metrics.csv'%(dataset_name))
            dataset_df = dataset_df[(dataset_df['n']%100)==0]
            print("BEF")
            print(dataset_df[dataset_df['algorithm']=='mmhc']['AIC'].mean(),dataset_df[dataset_df['algorithm']=='hc']['AIC'].mean())
            print(dataset_df[dataset_df['algorithm']=='mmhc']['BIC'].mean(),dataset_df[dataset_df['algorithm']=='hc']['BIC'].mean())
            print(len(dataset_df.index))
            dataset_df = dataset_df[dataset_df['algorithm']==algorithm]
            print(len(dataset_df.index))

            dataset_df = dataset_df[['dataset_name','n',score]]
            dataset_df['dataset_name']=dataset_name_to_printable_exp3[dataset_name]
            print(len(dataset_df.index))
            print("AFTER")
            dfs += [dataset_df]
            max_n  = dataset_df['n'].max()
            print(algorithm,score,dataset_name,np.mean(dataset_df[dataset_df['n']==max_n][score]))
        concat_df = pd.concat(dfs)

        concat_df = concat_df.rename(index=str, columns={"dataset_name": "Dataset",'n':'# data points'})
        concat_df = concat_df.rename(index=str, columns={"dataset_name": "Dataset",'n':'# data points','logLik':'Log Likelihood'})
        ax = sns.pointplot(x='# data points', y=(score if score!='logLik' else 'Log Likelihood'), hue='Dataset',data=concat_df)
        ax.set_title('Algorithm: %s, Metric: %s'%(algorithm,score if score!='logLik' else 'Log Likelihood'))
        fig = ax.get_figure()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig("/Users/Matt/Desktop/exp3_%s_%s.png" % (algorithm,score))
        plt.clf()


df = pd.read_csv(exp1_csv_folder+'logLik_scores.csv')
for score in ['BIC','Log Likelihood']:
    for algorithm in algorithms:
        algo_df = df[df['algorithm']==algorithm]
        algo_df = algo_df.rename(index=str, columns={"dataset": "Dataset",'nvars':'# variables','logLik':'Log Likelihood'})
        algo_df['Dataset'] = algo_df['Dataset'].map(dataset_name_to_printable_exp3)
        ax = sns.pointplot(x='# variables', y=score, hue='Dataset',data=algo_df)
        ax.set_title('Algorithm: %s, Metric: %s'%(algorithm,score))
        fig = ax.get_figure()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig("/Users/Matt/Desktop/exp3_byvars_%s_%s.png" % (algorithm,score))
        plt.clf()



df = pd.read_csv(exp1_csv_folder+'loss_scores.csv')

for algorithm in algorithms:
    algo_df = df[df['algorithm']==algorithm]
    algo_df = algo_df.rename(index=str, columns={"dataset": "Dataset",'logLik_loss':'Log Likelihood Loss','nvars':'# variables'})
    algo_df['Dataset'] = algo_df['Dataset'].map(dataset_name_to_printable_exp3)
    ax = sns.pointplot(x='# variables', y='Log Likelihood Loss', hue='Dataset',data=algo_df)
    ax.set_title('Algorithm: %s, Metric: %s'%(algorithm,"Log Loss"))
    fig = ax.get_figure()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig("/Users/Matt/Desktop/exp3_byvars_%s_%s.png" % (algorithm,'loss'))
    plt.clf()

df = pd.read_csv(exp1_csv_folder+'loss_scores_byn.csv')

for algorithm in algorithms:
    algo_df = df[df['algorithm']==algorithm]
    algo_df = algo_df.rename(index=str, columns={"dataset": "Dataset",'logLik_loss':'Log Likelihood Loss','n':'# data points'})
    algo_df['Dataset'] = algo_df['Dataset'].map(dataset_name_to_printable_exp3)
    ax = sns.pointplot(x='# data points', y='Log Likelihood Loss', hue='Dataset',data=algo_df)
    ax.set_title('Algorithm: %s, Metric: %s'%(algorithm,"Log Loss"))
    fig = ax.get_figure()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig("/Users/Matt/Desktop/exp3_byn_%s_%s.png" % (algorithm,'loss'))
    plt.clf()
