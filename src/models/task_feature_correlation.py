from __future__ import print_function
from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import offsetbox
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import metrics
import scipy.stats.mstats as mstats
import scipy.stats as stats
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline

import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import itertools
import scipy





#
#
# BEGIN GLOBALS
#
#




#
#
# END GLOBALS
#
#



#
#
# BEGIN FUNCTIONS
#
#

def jaccard(x,y):
    if len(set(x)|set(y)) == 0:
        return 0
    return float(len(set(x)&set(y)))/float(len(set(x)|set(y)))

def run_correlation(df):
    global FEATURES
    global P_SIGNIFICANT

    results = []



    for intention in INTENTION_COLUMNS:
        for feature in FEATURES:
            res = {'feature':feature,'intention':intention}
            group1 = df[df["intent_current_"+intention]==0][feature].tolist()
            group2 = df[df["intent_current_"+intention]==1][feature].tolist()


            are_norm = True
            (s,p) = mstats.normaltest(group1)
            are_norm = are_norm and (p > P_SIGNIFICANT)
            (s,p) = mstats.normaltest(group2)
            are_norm = are_norm and (p > P_SIGNIFICANT)
            if are_norm:
                (s,p) = stats.f_oneway(group1,group2)
                res['test'] = 'One-way ANOVA'
                res['statistic'] = s
                res['p'] = p
                res['mean_0'] = np.mean(group1)
                res['mean_1'] = np.mean(group2)
            else:
                (s,p) = mstats.kruskalwallis(group1,group2)
                res['test'] = 'Kruskal-Wallis'
                res['statistic'] = s
                res['p'] = p
                res['mean_0'] = np.mean(group1)
                res['mean_1'] = np.mean(group2)

            results += [res]
    return results


def train_test_split_list(l,train_test_split=0.33):
    yes = None
    no = None
    nelements=len(l)
    n_train = int(train_test_split*len(l))
    n_test = len(l)-n_train

    # print l,len(l),n_train
    yes = list(np.random.choice(l,n_train,replace=False))
    no = [e for e in l if not e in yes]
    assert sorted(yes+no)==sorted(l)
    return (yes,no)

#
#
# END FUNCTIONS
#
#


#TODO:
# Remove PERSONALIZED
# Handle split by userIDs, for personalized training/testing
if __name__=='__main__':

    PREDICTIONS = [
        # 'userID',
        'facet_trec2014tasktype',
        'facet_product',
        'facet_goal',
        # 'facet_goalplusproduct',
        # 'facet_complexity',
        # 'facet_complexity_merged',
        # 'facet_complexity_create'
    ]

    METADATA = [
        'facet_product', 'facet_goal', 'facet_trec2014tasktype',
        'userID',
        'Unnamed: 0', 'questionID', 'queryID', 'session_num',
        'interaction_num', 'action_type', 'local_timestamp',
        'data', 'start_time', 'start_action', 'total_actions']

    PERSONALIZED = False

    PREDICTION_VALUE_TO_ACRONYM = {
        'Relationships': 'REL',
        1: 'TRU',
        'Story Pitch': 'STP',
        'Factual': 'FAC',
        'all': 'ALL',
        'Amorphous': 'AMO',
        'Intellectual': 'INT',
        'Specific': 'SPE',
        0: 'FAL',
        'Copy Editing': 'CPE',
        'Coelacanths': 'COE',
        'Interview Preparation': 'INT',
        'Document': 'DOC',
        'Segment': 'SEG',
        'Methane clathrates and global warming': 'MTH'
    }

    FEATURESET_TO_ACRONYM = {
        'stratified': 'STR',
        'mostfrequent_baseline': 'MFQ',
        'bookmark': 'BK',
        'contentpage': 'CP',
        'contentandbookmark': 'CP_BK',
        'serp': 'SP',
        'query': 'QU',
        'serpandquery': 'SP_QU',
        'serp_session': 'SP_SESS',
        'query_session': 'QU_SESS',
        'intention': 'INT',
        'serpandquery_session': 'SP_QU_SESS',
        'allfeatures_segment': 'ALL_SEG',
        'allfeatures_session': 'ALL_SESS',
        'allfeatures': 'ALL',
        'allfeatures_default': 'ALL_DEF',
        'allfeatures_svc': 'ALL_SVC',
        'allfeatures_ada': 'ALL_ADA',
        'allfeatures_knn': 'ALL_KNN',
        'allfeatures_gnb': 'ALL_GNB',
        'allfeatures_ovr': 'ALL_OVR',
        'allfeatures_ovo': 'ALL_OVO',
        'allfeatures_mlp': 'ALL_MLP',
        'allfeatures_gsc': 'ALL_gsc',
    }

    INTENTION_COLUMN_TO_ACRONYM = {
        'userID': 'UID',
        'facet_trec2014tasktype': 'TASKTYPE',
        'facet_product': 'PRODUCT',
        'facet_level': 'LEVEL',
        'facet_goal': 'GOAL',
        'facet_named': 'NAMED',
        'facet_complexity': "COMPLEX",
        'facet_complexity_merged': "COMPLEX_MERGED",
        'facet_complexity_create': "COMPLEX_CREATE",
        'facet_goalplusproduct': "GOALPLUSPRODUCT",
        'facet_trec2014knownitem': 'KNOWNITEM',
        'facet_trec2014knownsubject': 'KNOWNSUBJECT',
        'facet_trec2014interpretive': 'INTERPRETIVE',
        'facet_trec2014exploratory': 'EXPLORATORY'
    }

    FACETVALUE_TO_SHORTNAME = {
        '': 'ALL',
        None: 'ALL',
        'Factual': 'prod_FACT',
        'Intellectual': 'prod_INTEL',
        'Document': 'level_DOC',
        'Segment': 'level_SEG',
        'Amorphous': 'goal_AMOR',
        'Specific': 'goal_SPEC',
        0: 'named_NO',
        1: 'named_YES',
        '0': 'named_NO',
        '1': 'named_YES',
        'Coelacanths': 'topic_COE',
        'Methane clathrates and global warming': 'topic_MET',
        'Copy Editing': 'type_CPE',
        'Interview Preparation': 'type_INT',
        'Story Pitch': 'type_STP',
        'Relationships': 'type_REL',
        'all': 'type_all'
    }





    dataset_lastqueryonly = pd.read_csv('../../data/processed/features_trec_lastquery.csv')
    dataset_firstqueryonly = pd.read_csv('../../data/processed/features_trec_firstquery.csv')


    FEATURES = [c for c in list(dataset_lastqueryonly.columns.values) if not c in PREDICTIONS + METADATA]



    RUN_CT = 0
    ALL_FEATURES = FEATURES

    featurenames_and_featuresets = [("stratified", ALL_FEATURES,'str'),
                                    ("mostfrequent_baseline", ALL_FEATURES,'mfq'),
                                    ("allfeatures_ada", ALL_FEATURES, 'ada'),
                                    ("allfeatures_knn", ALL_FEATURES, 'knn'),
                                    ("allfeatures_gnb", ALL_FEATURES, 'gnb'),
                                    # ("allfeatures_ovr", ALL_FEATURES, 'ovr'),
                                    # ("allfeatures_ovo", ALL_FEATURES, 'ovo'),
                                    ("allfeatures_mlp", ALL_FEATURES, 'mlp')
                                    ]

    lr_results = []
    result_dump_dataframe = None





    # # BEGIN TRIALS
    #TODO: All of this!
    # 1) Robust p-value testing
    # 2) homogeneity

    for (dfname, df) in [('firstquery', dataset_firstqueryonly), ('lastquery', dataset_lastqueryonly)]:
        N_TOTAL_TRIALS = 10000
        n_conducted_trials = 0

        trials_df = []
        n_significant = dict()
        for feature in FEATURES:
            for p in [.05, .01, .001]:
                n_significant[('facet_goal',feature,p)] = 0
                n_significant[('facet_product', feature, p)] = 0
                n_significant[('facet_trec2014tasktype', feature, p)] = 0


        task_types = list(set(df['facet_trec2014tasktype'].tolist()))

        users_onetask = []
        task_type_count = None
        tasks_assigned = []
        session_nums = []

        for (userID, group) in df.groupby(['userID']):
            task_type_count = Counter(group['facet_trec2014tasktype'].tolist())
            n_sessions = len(list(set(group['session_num'].tolist())))
            if len(task_type_count)==1:
                users_onetask+=[userID]
                tasks_assigned += [group['facet_trec2014tasktype'].tolist()[0]]
                session_nums += [group['session_num'].tolist()[0]]


        n_left_counter = Counter(tasks_assigned)
        for n in n_left_counter.keys():
            n_left_counter[n] = max(0, 65 - n_left_counter[n])


        # 194 chosen at this point, 66 are random
        print(session_nums)
        print(len(session_nums))
        print(len(list(set(session_nums))))
        print(n_left_counter)


        prediction_dataset_multipletasks = df[~df['userID'].isin(users_onetask)]

        while(n_conducted_trials < N_TOTAL_TRIALS):
            print("N CONDUCTED",n_conducted_trials)

            tasks_multipletasksusers = []
            sessions_multipletaskusers = []
            users_multipletasks_assigned = []

            valid = True


            data_usersleft = prediction_dataset_multipletasks[~prediction_dataset_multipletasks['userID'].isin(users_multipletasks_assigned)]

            for (tasktype,count) in n_left_counter.most_common(4):
                for _ in range(count):
                    session_nums_multipletaskusers = data_usersleft[data_usersleft['facet_trec2014tasktype']==tasktype]['session_num'].tolist()
                    if len(session_nums_multipletaskusers)==0:
                        valid = False
                        break
                    random_session = random.choice(session_nums_multipletaskusers)

                    assigned_user = data_usersleft[data_usersleft['session_num']==random_session]['userID'].tolist()[0]
                    sessions_multipletaskusers += [random_session]
                    users_multipletasks_assigned += [assigned_user]
                    tasks_multipletasksusers += [tasktype]
                    data_usersleft = prediction_dataset_multipletasks[
                        ~prediction_dataset_multipletasks['userID'].isin(users_multipletasks_assigned)]
                if valid == False:
                    break


            assert len(users_multipletasks_assigned) == len(set(users_multipletasks_assigned))
            total_counter = Counter(tasks_multipletasksusers)+Counter(tasks_assigned)
            if not valid:
                if max(total_counter.values())-min(total_counter.values())>3:
                    # print("FAILURE!",Counter(tasks_multipletasksusers) + Counter(tasks_assigned))
                    continue
                # else:
                #     print(users_multipletasks_assigned)


            assert len(sessions_multipletaskusers) + len(users_onetask) in [260,259,258,257,256]
            n_conducted_trials += 1


            statistical_significance_dataset = df[df['session_num'].isin(sessions_multipletaskusers + session_nums)]
            assert len(sessions_multipletaskusers) + len(users_onetask) == len(statistical_significance_dataset.index)

            for target in ['facet_trec2014tasktype', 'facet_goal', 'facet_product']:
                groups = []
                for (n, group) in statistical_significance_dataset.groupby(target):
                    groups += [group]
                for feature in FEATURES:
                    groups_featurevalues = [g[feature] for g in groups]
                    (stat,p) = (None,None)
                    if target =='facet_trec2014tasktype':
                        (stat1,pnorm1) = stats.normaltest(groups_featurevalues[0])
                        (stat2, pnorm2) = stats.normaltest(groups_featurevalues[1])
                        (stat3, pnorm3) = stats.normaltest(groups_featurevalues[2])
                        (stat4, pnorm4) = stats.normaltest(groups_featurevalues[3])
                        if(pnorm1 < .05 or pnorm2 < .05 or pnorm3 < .05 or pnorm4 < .05):
                            (stat,p) = stats.kruskal(groups_featurevalues[0],groups_featurevalues[1],groups_featurevalues[2],groups_featurevalues[3])
                        else:
                            print("normal ANOVA")
                            (stat, p) = stats.f_oneway(groups_featurevalues[0], groups_featurevalues[1],
                                                      groups_featurevalues[2], groups_featurevalues[3])
                    else:
                        (stat1, pnorm1) = stats.normaltest(groups_featurevalues[0])
                        (stat2, pnorm2) = stats.normaltest(groups_featurevalues[1])
                        if (pnorm1 < .05 or pnorm2 < .05):
                            (stat, p) = stats.mannwhitneyu(groups_featurevalues[0], groups_featurevalues[1])
                        else:
                            print("normal ttest")
                            (stat, p) = stats.ttest_ind(groups_featurevalues[0], groups_featurevalues[1])

                    trials_df += [{'n_trial': n_conducted_trials, 'target': target, 'feature': feature, 'pvalue': p}]
                    for pthreshold in [.05,.01,.001]:
                        n_significant[(target,feature,pthreshold)] += pthreshold > p



        df = []
        for (i,j,k) in n_significant.keys():
            df += [{'target':i,'feature':j,'p':k,'count':n_significant[(i,j,k)],'probability':n_significant[(i,j,k)]/float(N_TOTAL_TRIALS)}]

        print(pd.DataFrame(df).sort_values(['target','count'],ascending=[0,0]))
        pd.DataFrame(df).sort_values(['target', 'probability'], ascending=[0, 0]).to_csv('/Users/Matt/Desktop/significance_trials.csv')
        pd.DataFrame(trials_df).to_csv('/Users/Matt/Desktop/trials_dump.csv')
        exit()

        while n_conducted_trials < N_TOTAL_TRIALS:
            print("N CONDUCTED",n_conducted_trials)
            session_nums = []
            session_nums_onesession = []
            tasktype_count = None
            for (n,group) in prediction_dataset.groupby(['userID']):
                task_type_count = Counter(prediction_dataset[prediction_dataset['session_num'].isin(session_nums)]['facet_trec2014tasktype'].tolist())
                user_session_nums = list(set(group['session_num'].tolist()))
                task_types_present = []
                task_types_randomchoice = []
                for(n2,group2) in group.groupby(['facet_trec2014tasktype']):
                    count = task_type_count[n2]
                    weight = (260-count)/260
                    task_types_randomchoice += [n2]*int(weight*260)

                print(Counter(task_types_randomchoice))
                usertasktype = random.choice(task_types_randomchoice)
                usersessions = group[group['facet_trec2014tasktype']==usertasktype]['session_num'].tolist()
                session_nums += [random.choice(usersessions)]



                # session_nums += [random.choice(user_session_nums)]

                # if len(group.index)==1:
                #     session_nums_onesession += [session_nums[-1]]


            sessions = prediction_dataset[prediction_dataset['session_num'].isin(session_nums)]
            valid = True
            task_type_count = {}
            for task_type in task_types:
                n_sessions = len(sessions[sessions['facet_trec2014tasktype']==task_type].index)
                task_type_count[task_type] = n_sessions

            if max(task_type_count.values())-min(task_type_count.values()) >= 2:
                print(max(task_type_count.values())-min(task_type_count.values()))
                continue

            n_conducted_trials += 1

    exit()
    # END TRIALS







    # Possible TODO:
    # Number of correlations.
    # ki_data = prediction_dataset[prediction_dataset['facet_trec2014tasktype']=='known-item']
    # ks_data = prediction_dataset[prediction_dataset['facet_trec2014tasktype']=='known-subject']
    # interp_data = prediction_dataset[prediction_dataset['facet_trec2014tasktype']=='interpretive']
    # explor_data = prediction_dataset[prediction_dataset['facet_trec2014tasktype']=='exploratory']
    # spec_data = prediction_dataset[prediction_dataset['facet_goal']=='specific']
    # amor_data = prediction_dataset[prediction_dataset['facet_goal'] == 'amorphous']
    # fact_data = prediction_dataset[prediction_dataset['facet_product'] == 'factual']
    # intel_data = prediction_dataset[prediction_dataset['facet_product'] == 'intellectual']
    # user_data = []
    # for u in list(set(prediction_dataset['userID'].tolist())):
    #     user_data += [prediction_dataset[prediction_dataset['userID']==u]]
    #
    # n_features = len(FEATURES)
    # n_users = len(user_data)
    #
    # pval = .05
    #
    #
    # sig_correction_tasks = pval/scipy.misc.comb(4,2)
    # sig_correction_users = pval / scipy.misc.comb(len(user_data), 2)
    # n_task_sig = 0
    # n_user_sig = 0
    # n_goal_sig = 0
    # n_prod_sig = 0
    # n_task_sig_bonferr = 0
    # for attribute in FEATURES:
    #     print(attribute, np.mean(ki_data[attribute]),np.mean(ks_data[attribute]),np.mean(interp_data[attribute]),np.mean(explor_data[attribute]))
    #     print(attribute, np.mean(spec_data[attribute]),np.mean(amor_data[attribute]))
    #     print(attribute, np.mean(fact_data[attribute]),np.mean(intel_data[attribute]))
    #     (h,p) = stats.kruskal(ki_data[attribute], ks_data[attribute], interp_data[attribute],
    #                   explor_data[attribute])
    #     # print((h,p))
    #     n_task_sig +=  p < pval
    #
    #     for (d1,d2) in itertools.combinations([ki_data[attribute], ks_data[attribute], interp_data[attribute],
    #                   explor_data[attribute]],2):
    #         (h,p) = stats.mannwhitneyu(d1,d2)
    #         n_task_sig_bonferr += p < sig_correction_tasks
    #
    #
    #
    #     (h, p) = stats.mannwhitneyu(spec_data[attribute], amor_data[attribute])
    #     n_goal_sig += p < pval
    #     (h, p) = stats.mannwhitneyu(fact_data[attribute],intel_data[attribute])
    #     n_prod_sig += p < pval
    #     (h,p) = stats.kruskal(*[u[attribute] for u in user_data])
    #     n_user_sig += p < pval
    #
    # n_errors = 0
    # n_personal_sig = 0
    # for attribute in FEATURES:
    #     for u in user_data:
    #         # print(u)
    #         try:
    #             ki_data_u = u[u['facet_trec2014tasktype'] == 'known-item']
    #             ks_data_u = u[u['facet_trec2014tasktype'] == 'known-subject']
    #             interp_data_u = u[u['facet_trec2014tasktype'] == 'interpretive']
    #             explor_data_u = u[u['facet_trec2014tasktype'] == 'exploratory']
    #             (h, p) = stats.kruskal(ki_data_u[attribute], ks_data_u[attribute], interp_data_u[attribute],
    #                                    explor_data_u[attribute])
    #             n_personal_sig += p < pval
    #             if p < pval:
    #                 print(u['userID'].tolist()[0],attribute)
    #         except ValueError:
    #             n_errors += 1
    #
    # print("N ERRORS",n_errors,"OUTOF",(n_features*n_users))
    # # print(n_task_sig,n_task_sig/n_features,n_user_sig,n_user_sig/(n_features*n_users))
    # print(n_task_sig,n_task_sig/n_features,n_user_sig,n_user_sig/n_features)
    # print("PERSONAL",n_personal_sig,n_features*n_users,n_personal_sig/(n_features*n_users))
    # # print(n_task_sig_bonferr,n_task_sig_bonferr/n_features)
    #
    # print("BINARY",n_goal_sig,n_goal_sig/n_features,n_prod_sig,n_prod_sig/n_features)
    # print("FEATURES",FEATURES)
    # exit()


    # for attribute in FEATURES:
    #     print(attribute+"-TASK",stats.kruskal(ki_data[attribute], ks_data[attribute],
    #                         interp_data[attribute], explor_data[attribute]),.05/scipy.misc.comb(4,2))
    #     print(attribute+"-USER",stats.kruskal(*[u[attribute] for u in user_data]),.05/scipy.misc.comb(len(user_data),2))
    # exit()