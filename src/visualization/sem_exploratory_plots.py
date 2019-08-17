import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools


# 1) binary vs. ordinal: boxplot
# 2) binary vs. continuous: boxplot
# 3) ordinal vs. continuous: boxplot
# 3) continuous vs continuous: scatterplot

# binary vs. binary
# ordinal vs ordinal
# binary: how do I show a regular distribution?
# ordinal: (histogram?) how do I show a regular distribution?
# continuous: how do I show a regular distribution?

eopn_data_location = '/Users/Matt/Desktop/features_eopn_alldata.csv'
eopn_df = pd.read_csv(eopn_data_location)
eopn_binary_variables = ['facet_goal','facet_product']
eopn_ordinal_variables = ['actual_difficulty']
eopn_continuous_variables = ['queries_num','pages_per_query']


# sal_data_location = '/Users/Matt/Desktop/features_sal_alldata.csv'
# sal_df = pd.read_csv(sal_data_location)
# sal_binary_variables = ['task_goal','task_product']
# sal_ordinal_variables = ['post_difficulty']
# sal_continuous_variables = ['queries_num']

# nsf_data_location = '/Users/Matt/Desktop/features_nsf_alldata.csv'
# nsf_df = pd.read_csv(nsf_data_location)
# nsf_binary_variables = ['task_goal','task_product']
# nsf_ordinal_variables = ['post_difficulty']
# nsf_continuous_variables = ['num_queries']


outfolder = '/Users/Matt/Desktop/sem_exploratory_plots/'


for (dataname,df,binary_variables,ordinal_variables,continuous_variables) in [('eopn',eopn_df,eopn_binary_variables,eopn_ordinal_variables,eopn_continuous_variables),
# ('sal',sal_df)
           ]:
    # 1) binary vs. ordinal: boxplot
    for bv in binary_variables:
        for ov in ordinal_variables:
            ax = sns.boxplot(x=bv, y=ov, data=df)
            fig = ax.get_figure()
            fig.savefig(outfolder + '/%s/plot_%s_vs_%s.png' % (dataname,bv,ov))
            plt.clf()

    # 2) binary vs. continuous: boxplot
    for bv in binary_variables:
        for cv in continuous_variables:
            ax = sns.boxplot(x=bv, y=cv, data=df)
            fig = ax.get_figure()
            fig.savefig(outfolder + '/%s/plot_%s_vs_%s.png' % (dataname,bv,cv))
            plt.clf()

    # 3) ordinal vs. continuous: boxplot
    for ov in ordinal_variables:
        for cv in continuous_variables:
            ax = sns.boxplot(x=ov, y=cv, data=df)
            fig = ax.get_figure()
            fig.savefig(outfolder + '/%s/plot_%s_vs_%s.png' % (dataname,ov,cv))
            plt.clf()

    # 3) continuous vs continuous: scatterplot
    for (cv1,cv2) in itertools.combinations(continuous_variables,2):
        ax = sns.boxplot(x=cv1, y=cv2, data=df)
        fig = ax.get_figure()
        fig.savefig(outfolder + '/%s/plot_%s_vs_%s.png' % (dataname,cv1,cv2))
        plt.clf()