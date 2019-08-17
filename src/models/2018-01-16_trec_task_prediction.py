from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.stats.mstats as mstats
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.manifold import Isomap
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

DOT_DATA = None
#
#
# BEGIN GLOBALS
#
#

RUN_CT = 0
PERSONALIZED = False

PREDICTIONS = [
    'facet_trec2014tasktype',
    'facet_product',
    'facet_goal'
]


PREDICTIONS_TREC = [
    'facet_trec2014tasktype',
    'facet_product',
    'facet_goal',
]


PREDICTIONS_NSF = [
    'facet_trec2014tasktype',
    'facet_product',
    'facet_goal',
    'facet_level',
    'facet_named',
]


METADATA=[
    'facet_product','facet_goal','facet_trec2014tasktype',
    'userID',
    'Unnamed: 0','questionID','queryID','session_num','interaction_num','action_type','local_timestamp','data','start_time','start_action','total_actions_count'
    ]

FEATURES = []



FEATURESET_TO_ACRONYM = {
'stratified':'STR',
'mostfrequent_baseline':'MFQ',
'bookmark':'BK',
'contentpage':'CP',
'contentandbookmark':'CP_BK',
'serp':'SP',
'query':'QU',
'serpandquery':'SP_QU',
'serp_session':'SP_SESS',
'query_session':'QU_SESS',
'intention':'INT',
'serpandquery_session':'SP_QU_SESS',
'allfeatures_segment':'ALL_SEG',
'allfeatures_session':'ALL_SESS',
'allfeatures':'ALL',
'allfeatures_default':'ALL_DEF',
'allfeatures_svc':'ALL_SVC',
'allfeatures_ada':'ALL_ADA',
'allfeatures_knn':'ALL_KNN',
'allfeatures_gnb':'ALL_GNB',
'allfeatures_ovr':'ALL_OVR',
'allfeatures_ovo':'ALL_OVO',
'allfeatures_mlp':'ALL_MLP',
'allfeatures_gsc':'ALL_GSC',
'allfeatures_dct':'ALL_DCT',
'allfeatures_rfc':'ALL_RFC',
}




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

def run_correlation(df,feature,outcome):
    # print("FEATURE",feature,"OUTCOME",outcome)
    # print(len(df.index))
    P_SIGNIFICANT = .05

    outcomes = set(df[outcome].tolist())
    n_outcomes = len(outcomes)
    # print("N OUTCOMES",n_outcomes)

    groups = []
    for oc in outcomes:
        groups += [df[df[outcome]==oc][feature].tolist()]



    are_norm = True
    for g in groups:
        # print(g,len(g))
        (s,p) = mstats.normaltest(g)
        are_norm = are_norm and (p > P_SIGNIFICANT)

    result = {}
    if are_norm:
        if n_outcomes <=2:
            (s, p) = stats.ttest_ind(groups[0],groups[1])
            result['test'] = 't-test'
        else:
            (s,p) = stats.f_oneway(*groups)
            result['test'] = 'One-way ANOVA'
        result['statistic'] = s
        result['p'] = p
        for (n,g) in zip(range(len(groups)),groups):
            result['mean_%d'%n] = np.mean(g)
    else:
        if n_outcomes <=2:
            (s, p) = stats.mannwhitneyu(groups[0],groups[1])
            result['test'] = 'Mann-Whitney'
        else:
        # print(len(groups),len(groups[0]))
            (s,p) = mstats.kruskalwallis(*groups)
            result['test'] = 'Kruskal-Wallis'
        result['statistic'] = s
        result['p'] = p
        for (n,g) in zip(range(len(groups)),groups):
            result['mean_%d' % n] = np.mean(g)

    return result


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


DCT_COUNT = 0

def run_prediction(df,input_features=None,clf=None,num_iters=40):

    global FEATURES
    global RUN_CT
    global PREDICTIONS
    global METADATA
    global DCT_COUNT
    features = None


    print(input_features)
    if input_features is None or input_features==['dummy1'] or input_features==['dummy2']:
        features = [c for c in list(df.columns.values) if not c in PREDICTIONS + METADATA]
        # features = FEATURES
        # print "FEATURES",FEATURES[92:]
    else:
        features = input_features



    le = preprocessing.LabelEncoder()


    # print("input features",features)

    # scaler = Normalizer()
    scaler = StandardScaler()
    results = []


    df_copy = df.copy()

    userIDs = [1]

    if 'userID' in df_copy.columns.values:
        userIDs = set(df_copy['userID'].tolist())

    if not PERSONALIZED:
        userIDs = [1]

    for facet in PREDICTIONS:
        # if facet not in df.columns.values:
        #     continue
        # print(facet)
        for i in range(num_iters):
            # print(i)
            RUN_CT += 1
            res = {'prediction':facet}

            # 0) Slice to only feature columns and  test column
            # 0a) Convert data to np array
            # 1) Random permutation



            y_tests_all = []
            y_preds_all = []
            y_scores_all = []
            for u in userIDs:
                data_okay = True
                res = {'prediction': facet}
                # print(u, userIDs)
                if input_features not in ['dummy1','dummy2'] and facet!='userID' and PERSONALIZED:
                    df = df_copy[df_copy['userID'] == u]
                while True:


                    try:
                        (session_nums_train,session_nums_test) = train_test_split_list(list(set(df['session_num'].tolist())),0.8)
                        y = df[facet].as_matrix()
                        le.fit(y)

                        df_train = df[df['session_num'].isin(session_nums_train)]
                        df_test = df[df['session_num'].isin(session_nums_test)]
                        X_train = df_train[features].as_matrix()
                        X_test = df_test[features].as_matrix()
                        assert (len(df_train) + len(df_test))==len(df.index)
                        y_train = df_train[facet].tolist()

                        y_train = le.transform(y_train)
                        y_test = df_test[facet].tolist()
                        y_test = le.transform(y_test)
                        if len(set(y_train))==1 or len(set(y_test))==0:
                            data_okay = False
                            break


                        scaler = StandardScaler()

                        if input_features==['dummy1']:
                            lr_model = DummyClassifier(strategy='stratified',random_state=random.randint(1,4294967294))
                        elif input_features==['dummy2']:
                            lr_model = DummyClassifier(strategy='most_frequent',random_state=random.randint(1,4294967294))
                        else:

                            X_tofit = df[features].as_matrix()
                            y_tofit = df[facet].as_matrix()
                            y_tofit = le.transform(y_tofit)
                            scaler.fit(X_tofit)
                            X_tofit = scaler.transform(X_tofit)


                            imap = Isomap(n_components=10)
                            pca = PCA(n_components=10)
                            rfe = RFE(LinearSVC(random_state=random.randint(1, 4294967294)))
                            sfm = SelectFromModel(LinearSVC(random_state=random.randint(1, 4294967294)))



                            knn = KNeighborsClassifier(n_neighbors=10)
                            gnb = GaussianNB()
                            mlp = MLPClassifier(alpha=1)
                            gsc = GridSearchCV(LinearSVC(),{'dual':[True,False],'C':[0.1,1, 10]})
                            svm = SVC(probability=True, gamma=0.7, C=1, random_state=random.randint(1, 4294967294))
                            dct = DecisionTreeClassifier(max_depth=8)
                            rfc = RandomForestClassifier(random_state=random.randint(1, 4294967294))
                            ada = AdaBoostClassifier(random_state=random.randint(1, 4294967294))
                            qda = QuadraticDiscriminantAnalysis()
                            anovakbest_filter = SelectKBest(f_classif, k=min([20, len(features)]))
                            ovr = OneVsRestClassifier(LinearSVC(random_state=random.randint(1, 4294967294)))
                            ovo = OneVsOneClassifier(LinearSVC(random_state=random.randint(1, 4294967294)))
                            # ovr = OneVsRestClassifier(MLPClassifier(alpha=1))
                            # ovo = OneVsOneClassifier(MLPClassifier(alpha=1))

                            clf_map={'knn': knn,
                                     'gnb': gnb,
                                     'svm': svm,
                                     'rfc': rfc,
                                     'ada': ada,
                                     'qda': qda,
                                     'svc': svm,
                                     'mlp': mlp,
                                     'ovr':ovr,
                                     'ovo':ovo,
                                     'gsc':gsc,
                                     'dct':dct
                                     }
                            if clf is not None and clf in clf_map.keys():
                                # lr_model = clf_map[clf]
                                classifier = clf_map[clf]
                                lr_model = Pipeline([('anova', anovakbest_filter), ('clf', classifier)])
                                # if classifier==dct:
                                #     DOT_DATA = tree.export_graphviz(clf, out_file=None,
                                #                                     feature_names=input_features,
                                #                                     class_names=facet,
                                #                                     filled=True, rounded=True,
                                #                                     special_characters=True)

                                # lr_model = Pipeline([('pca', pca), ('clf', clf_map[clf])])
                                # lr_model = Pipeline([('feature_selection', rfe), ('clf', clf_map[clf])])
                                # lr_model = Pipeline([('feature_selection', sfm), ('clf', clf_map[clf])])
                            else:
                                lr_model = Pipeline([('anova', anovakbest_filter), ('clf', gnb)])




                        scaler.fit(X_train)
                        lr_model.fit(scaler.transform(X_train),y_train)
                        # if clf == 'dct':
                        #     DCT_COUNT += 1
                        #     export_graphviz(classifier,out_file='/Users/Matt/Desktop/output/out%d.dot'%DCT_COUNT,feature_names=features)
                        break
                    except ValueError as e:
                        print(e)
                        print("fail")
                        pass

                # print("INPUT FEATURES", input_features,u,facet)

                if not data_okay:
                    print("NOT OKAY!")
                    continue
                X_test = scaler.transform(X_test)
                y_pred = lr_model.predict(X_test)


                def f(label,l):
                    return sum(l==label)/len(l)

                # y_score = lr_model.predict_proba(X_test)[:,1]


                # y_score = lr_model.predict_proba(X_test)[:,1]
                y_tests_all += list(y_test)
                y_preds_all += list(y_pred)
                # y_scores_all += list(y_score)
            res["accuracy"] = metrics.accuracy_score(y_tests_all, y_preds_all)
            # res["f1"] = metrics.f1_score(y_test,y_pred,average='samples')
            # res["precision"] = metrics.precision_score(y_test,y_pred,average='samples')
            # res["recall"] = metrics.recall_score(y_test,y_pred,average='samples')
            # res["n_queries"] = df_test['queries_num'].tolist()

            # try:
            #     res["aucroc"] = metrics.roc_auc_score(y_test,y_score)
            # except ValueError:
            #     if sum(y_test) > 1:
            #         res["aucroc"] = 1
            #     else:
            #         res["aucroc"] = 0
            #
            # res["ap"] = metrics.average_precision_score(y_test,y_score)
            res['y_true'] = le.inverse_transform(y_tests_all)
            res['y_pred'] = le.inverse_transform(y_preds_all)
            # res['y_true'] = y_test
            # res['y_pred'] = y_pred
            # res['y_score'] = y_scores_all
            res['run_ct'] = RUN_CT

            # print "SCORE",res["f1"]

            # print "SCORE",res["ap"],res["aucroc"]

            results += [res]
    return pd.DataFrame(results)



#####
#
# BEGIN PREDICTION RESULTS
#
#####

#
#
# END FUNCTIONS
#
#




    # FEATURES = [
    #     'elapsed_minutes',
    #     'queries_num_session',
    #     'noclicks_queries_session',
    #     'query_lengths_mean',
    #     'query_lengths_mean_nonstop',
    #     'pages_num_session',
    #     'perquery_pages',
    #
    #     'queryterms_numunique_session',
    #
    #     'serps_num_session',
    #     'perquery_serps',
    #     'dwelltimes_percent_serp_session',
    #     'perquery_dwelltimes_total_serp_session',
    #
    #     'dwelltimes_mean_content_session',
    #     'dwelltimes_mean_serp_session',
    #     'dwelltimes_total_content_session',
    #     'dwelltimes_total_serp_session',
    #
    #     'total_actions_count',
    #
    #
    #     # elapsed_seconds
    # ]

    # FEATURES = ['elapsed_minutes','pages_num_session','queries_num_session','serps_num_session','dwelltimes_percent_serp_session','perquery_serps','perquery_dwelltimes_total_serp_session']



if __name__=='__main__':
    # LEQ = True
    LEQ = False

    PERSONALIZED = True

    #Step 1) Create several data frames

    # dataset = pd.read_csv('/Users/Matt/Desktop/sigir_features_nsf.csv')
    dataset = pd.read_csv('/Users/Matt/Desktop/sigir_features_trec.csv')
    FEATURES = [c for c in list(dataset.columns.values) if not c in PREDICTIONS + METADATA]



    #Whole session mean
    # df = pd.read_csv('/Users/Matt/Desktop/sigir_features_trec_wholesessionmean.csv')
    # df = df[df['start_action'] == False]
    # wholesessionmean_dataset = []
    # for (n,group) in df.groupby(['session_num']):
    #     wholesessionmean_dataset += [group.tail(1)]
    # wholesessionmean_dataset = pd.concat(wholesessionmean_dataset)
    #
    # #Whole session total
    # df = pd.read_csv('/Users/Matt/Desktop/sigir_features_trec_wholesessiontotal.csv')
    # df = df[df['start_action'] == False]
    # wholesessiontotal_dataset = []
    # for (n, group) in df.groupby(['session_num']):
    #     wholesessiontotal_dataset += [group.tail(1)]
    # wholesessiontotal_dataset = pd.concat(wholesessiontotal_dataset)
    #
    # firstquery_dataset = dataset.copy(deep=True)
    # firstquery_dataset = firstquery_dataset[firstquery_dataset['queries_num_session']==1]
    #
    # firstaction_dataset = dataset.copy(deep=True)
    # firstaction_dataset = firstaction_dataset[firstaction_dataset['total_actions_count']==1]



    # dataset_allsteps = []
    # for (n,group) in dataset[dataset['start_action'] == False].groupby(['session_num', 'queries_num_session']):
    #     dataset_allsteps += [group.tail(1)]
    # dataset_allsteps=pd.concat(dataset_allsteps)
    # ds_allsteps = []
    # for (n, group) in dataset_allsteps.groupby(['session_num']):
    #     max_n_queries = group['queries_num_session'].max()
    #     group['queries_percent_session'] = group['queries_num_session'] / float(max_n_queries)
    #     ds_allsteps += [group]
    # dataset_allsteps = pd.concat(ds_allsteps)


    dataset_tail = []
    dataset_endsteponly = dataset[dataset['start_action']==False]
    for (n, group) in dataset_endsteponly.groupby(['session_num']):
        dataset_tail += [group.tail(1)]
    dataset_tail = pd.concat(dataset_tail)

    # firstquery_dataset_tail = []
    # for (n, group) in firstquery_dataset[firstquery_dataset['start_action'] == False].groupby(['session_num']):
    #     firstquery_dataset_tail += [group.tail(1)]
    # firstquery_dataset_tail = pd.concat(firstquery_dataset_tail)
    #
    # firstaction_dataset_tail = []
    # for (n, group) in firstaction_dataset[firstaction_dataset['start_action'] == False].groupby(['session_num']):
    #     firstaction_dataset_tail += [group.tail(1)]
    # firstaction_dataset_tail = pd.concat(firstaction_dataset_tail)





    # counts_df = []
    # for (sn, group) in dataset_endsteponly.groupby(['session_num']):
    #     tasktype = group['facet_trec2014tasktype'].tolist()[0]
    #     product = group['facet_product'].tolist()[0]
    #     goal = group['facet_goal'].tolist()[0]
    #     counts_df += [{'session_num':sn,'type':tasktype,'product':product,'goal':goal}]
    # counts_df = pd.DataFrame(counts_df)




    # query_count = dict()
    # action_count = dict()
    # for ((sn,qn),group) in dataset_endsteponly.groupby(['session_num','queries_num_session']):
    #     query_count[qn] = query_count.get(qn,0) + 1
    # for ((sn,an),group) in dataset_endsteponly.groupby(['session_num','total_actions_count']):
    #     action_count[an] = action_count.get(an,0) + 1

    # query_count
    # {1: 1021,
    #  2: 765,
    #  3: 555,
    #  4: 368,
    #  5: 225,
    #  6: 119,
    #  7: 70,
    #  8: 35,
    #  9: 16,
    #  10: 12,
    #  11: 7,
    #  12: 5,
    #  13: 4,
    #  14: 2,
    #  15: 1}

    # action_count
    # {1: 1021,
    #  2: 906,
    #  3: 789,
    #  4: 668,
    #  5: 521,
    #  6: 377,
    #  7: 282,
    #  8: 186,
    #  9: 119,
    #  10: 70,
    #  11: 46,
    #  12: 30,
    #  13: 19,
    #  14: 13,
    #  15: 11,
    #  16: 9,
    #  17: 8,
    #  18: 6,
    #  19: 5,
    #  20: 3,
    #  21: 3,
    #  22: 2}

    # max_n_queries_session = 15
    # qnum_range = [i for i in range(max_n_queries_session,0,-1) if query_count[i] >= 40]
    # anum_range = [i for i in action_count.keys() if action_count[i] >= 40]
    # dataset_endsteponly_bynqueries = dict()
    # dataset_endsteponly_bynactions = dict()


    # for i in qnum_range:
    #     if LEQ:
    #         dataset_endsteponly_bynqueries[i] = []
    #         df = dataset_tail.copy(deep=True)
    #         df = df[df['start_action'] == False]
    #         df = df[df['queries_num_session'] <= i]
    #         for (n, group) in df.groupby(['session_num']):
    #             dataset_endsteponly_bynqueries[i] += [group.tail(1)]
    #         dataset_endsteponly_bynqueries[i] = pd.concat(dataset_endsteponly_bynqueries[i])
    #     else:
    #         dataset_endsteponly_bynqueries[i] = []
    #         df = dataset.copy(deep=True)
    #         df = df[df['start_action']==False]
    #         df = df[df['queries_num_session'] == i]
    #         for (n,group) in df.groupby(['session_num']):
    #             dataset_endsteponly_bynqueries[i] += [group.tail(1)]
    #         dataset_endsteponly_bynqueries[i] = pd.concat(dataset_endsteponly_bynqueries[i])

    # for i in anum_range:
    #     if LEQ:
    #         dataset_endsteponly_bynactions[i] = []
    #         df = dataset_tail.copy(deep=True)
    #         df = df[df['start_action'] == False]
    #         df = df[df['total_actions_count'] <= i]
    #         for (n, group) in df.groupby(['session_num']):
    #             dataset_endsteponly_bynactions[i] += [group.tail(1)]
    #         dataset_endsteponly_bynactions[i] = pd.concat(dataset_endsteponly_bynactions[i])
    #     else:
    #         dataset_endsteponly_bynactions[i] = []
    #         df = dataset.copy(deep=True)
    #         df = df[df['start_action'] == False]
    #         df = df[df['total_actions_count'] == i]
    #         for (n, group) in df.groupby(['session_num']):
    #             dataset_endsteponly_bynactions[i] += [group.tail(1)]
    #         dataset_endsteponly_bynactions[i] = pd.concat(dataset_endsteponly_bynactions[i])




    # if LEQ:
    #     print(dataset_endsteponly_bynqueries.keys())
    #     print(dataset_endsteponly_bynqueries[4]['queries_num_session'].tolist())
    #     exit()



    sessions_per_user_df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+'/../../data/interim/trec2014_sessions_per_user_nonnegativeuserid.csv')
    active_users = list(set(sessions_per_user_df[sessions_per_user_df['num_sessions'] >=40]['user_num'].tolist()))
    # print("N USERS",len(list(set(active_users))))
    #
    #
    # users_per_topic_df = pd.read_csv('../../data/interim/trec2014_users_per_topic_nonnegativeuserid.csv')
    # active_topics = list(set(users_per_topic_df[users_per_topic_df['num_sessions'] >= 20]['topic_num'].tolist()))
    #
    # sessions_overview_df = pd.read_csv('../../data/interim/trec2014_sessions_overview.csv')
    # short_sessions = list(set(sessions_overview_df[sessions_overview_df['queries_num_session'] <= 3]['session_num'].tolist()))




    dataset_endsteponly_activeusersonly = dataset_endsteponly[dataset_endsteponly['userID'].isin(active_users)]

    featurenames_and_featuresets = [
                                    ("stratified", ['dummy1'], 'str'),
                                    ("mostfrequent_baseline", ['dummy2'], 'mfq'),
                                    ("allfeatures_ada", FEATURES, 'ada'),
                                    # ("allfeatures_knn", FEATURES, 'knn'),
                                    ("allfeatures_gnb", FEATURES, 'gnb'),
                                    # ("allfeatures_ovr", FEATURES, 'ovr'),
                                    # ("allfeatures_ovo", FEATURES, 'ovo'),
                                    ("allfeatures_svc", FEATURES, 'svc'),
                                    # ("allfeatures_rfc", FEATURES, 'rfc'),
                                    # ("allfeatures_mlp", FEATURES, 'mlp'),
                                    ("allfeatures_dct", FEATURES, 'dct')
                                    ]

    # nsf_firstquery = pd.read_csv('/Users/Matt/Desktop/nsf_first_query.csv')
    # nsf_wholesession = pd.read_csv('/Users/Matt/Desktop/nsf_whole_session.csv')


    # for c in firstquery_dataset_tail.columns.values:
    #     if c not in METADATA and c in ['perquery_pages',
    #
    #                                    'query_lengths_range',
    #                                    'dwelltimes_mean_content_session',
    #                                    'perquery_dwelltimes_total_content_session',
    #                                    'perquery_dwelltimes_total_serp_session',
    #                                    'serps_num_session',
    #                                    'queries_num_session']:
    #         firstquery_dataset_tail = firstquery_dataset_tail.drop(c,1)


    # for c in nsf_firstquery.columns.values:
    #     if c not in METADATA and c in ['time_dwelltotal_mean_content_segment', 'time_dwelltotal_mean_serp_segment']:
    #         nsf_firstquery = nsf_firstquery.drop(c,1)



    for c in dataset_tail.columns.values:
        if c not in METADATA and c in ['serps_num_session']:
            dataset_tail = dataset_tail.drop(c,1)
            dataset_endsteponly_activeusersonly.drop(c,1)


    # for c in nsf_wholesession.columns.values:
    #     if c not in METADATA and c in []:
    #         nsf_wholesession = nsf_wholesession.drop(c,1)


    # pprint.pprint([c for c in firstquery_dataset_tail.columns.values if not c in PREDICTIONS+METADATA])
    # pprint.pprint([c for c in nsf_firstquery.columns.values if not c in PREDICTIONS+METADATA])

    # pprint.pprint([c for c in dataset_tail.columns.values if not c in PREDICTIONS + METADATA])
    # pprint.pprint([c for c in nsf_wholesession.columns.values if not c in PREDICTIONS + METADATA])
    # exit()



    dataset_list = [
        ('trec_allsteps',dataset_endsteponly_activeusersonly),
        # ('trec_wholesession', dataset_tail),
                    # ('trec_wholesessionmean',wholesessionmean_dataset),
                    # ('trec_wholesessiontotal',wholesessiontotal_dataset),
                    # ('trec_firstquery', firstquery_dataset_tail),
                    # ('nsf_firstquery', nsf_firstquery),
                    # ('nsf_wholesession', nsf_wholesession),
                    ]

    # for i in qnum_range:
    #     dataset_list += [('session_query%d'%i,dataset_endsteponly_bynqueries[i])]

    # for i in anum_range:
    #     # if True:
    #     # if i%3==1:
    #     if i <=10:
    #         dataset_list += [('session_action%d' % i, dataset_endsteponly_bynactions[i])]


    prediction_results_all = []
    prediction_results_all_ci = []

    prediction_accuracy_results = dict()
    prediction_results_dump = []
    for (dfname, df) in dataset_list:
        features_ds = [c for c in list(df.columns.values) if not c in PREDICTIONS + METADATA]
        print("DFNAME", dfname, "NROWS", len(df.index), features_ds)
        # print("DFNAME",dfname,"NROWS",len(df.index),"NQUERIES",max(df['queries_num_session'].tolist()),"NACTIONS",max(df['total_actions_count'].tolist()))
        # for (dfname,df) in [('trec_firstquerysegment',firstquery_dataset_tail)]:
        #Get correlation results, output to pivot
        correlation_results = []
        for f in features_ds:
            for p in PREDICTIONS:
                # if p not in df.columns.values:
                #     continue
                correlation_results += [{'feature':f,'outcome':p,'correlation':run_correlation(df,f,p)['p']}]


        if LEQ:
            pd.DataFrame(correlation_results).pivot(index='feature', columns='outcome', values='correlation').to_csv(
                '/Users/Matt/Desktop/output/%s_correlation_leq.csv' % dfname)
        else:
            pd.DataFrame(correlation_results).pivot(index='feature',columns='outcome',values='correlation').to_csv('/Users/Matt/Desktop/output/%s_correlation.csv'%dfname)

        #Get average prediction results, output to pivot
        prediction_results = []


        for (featuregroupname, featuregroup, clf) in featurenames_and_featuresets:
            if featuregroup != ['dummy1'] and featuregroup != ['dummy2']:
                featuregroup = features_ds

            print(featuregroupname)
            lr_res = run_prediction(df.copy(), input_features=featuregroup, clf=clf,num_iters=40)
            prediction_results_dump += [lr_res]

            clf_rename = clf
            if clf_rename in ['str', 'mfq']:
                clf_rename = "base_" + clf_rename
            prediction_results_dump[-1]['classifier'] = clf_rename
            prediction_results_dump[-1]['dfname'] = dfname
            prediction_results_dump[-1]['featuregroupname'] = featuregroupname



            # if clf=='dct':
            #     graph = graphviz.Source(DOT_DATA)
            #     print(graph)
            for (p,group) in lr_res.groupby(['prediction']):
                print("PREDICTION",p)
                av_res = group['accuracy'].mean()

                prediction_accuracy_results[(dfname,clf,p)] = group['accuracy'].tolist()


                clf_rename = clf
                if clf_rename in ['str','mfq']:
                    clf_rename = "base_"+clf_rename


                if 'session_query' in dfname:
                    prediction_results += [{'classifier': clf_rename, 'prediction': p, 'accuracy': av_res, 'queries_num_session':max(df['queries_num_session'].tolist())}]
                    prediction_results_all += [{'classifier': clf_rename, 'prediction': p, 'accuracy': av_res, 'queries_num_session':max(df['queries_num_session'].tolist())}]
                    prediction_results_all_ci += [group.copy(deep=True)]
                    prediction_results_all_ci[-1]['queries_num_session'] = max(df['queries_num_session'].tolist())
                    prediction_results_all_ci[-1]['classifier'] = clf_rename
                elif 'session_action' in dfname:
                    prediction_results += [{'classifier': clf_rename, 'prediction': p, 'accuracy': av_res, 'total_actions_count':max(df['total_actions_count'].tolist())}]
                    prediction_results_all += [{'classifier': clf_rename, 'prediction': p, 'accuracy': av_res, 'total_actions_count':max(df['total_actions_count'].tolist())}]
                    prediction_results_all_ci += [group.copy(deep=True)]
                    prediction_results_all_ci[-1]['total_actions_count'] = max(df['total_actions_count'].tolist())
                    prediction_results_all_ci[-1]['classifier'] = clf_rename

                else:
                    prediction_results += [{'classifier': clf_rename, 'prediction': p, 'accuracy': av_res}]



        if ('session_query' in dfname) or ('session_action' in dfname):
            print(list(pd.DataFrame(prediction_results_all).columns.values))
            print("NNNQUERIES",set(pd.DataFrame(prediction_results_all)['queries_num_session'].tolist()))

        print(prediction_results)
        if LEQ:
            pd.DataFrame(prediction_results).pivot(index='classifier', columns='prediction', values='accuracy').to_csv(
                '/Users/Matt/Desktop/output/%s_prediction_leq.csv' % dfname)
        else:
            pd.DataFrame(prediction_results).pivot(index='classifier', columns='prediction', values='accuracy').to_csv('/Users/Matt/Desktop/output/%s_prediction.csv'%dfname)

    prediction_results_dump = pd.concat(prediction_results_dump)
    for (dfname,df) in dataset_list:
        for p in PREDICTIONS:
            df_toplot = prediction_results_dump[(prediction_results_dump['dfname'] == dfname) & (prediction_results_dump['prediction'] == p)]
            print(df_toplot)
            print(df_toplot.columns.values)
            plt.ylim(0,1.0)
            sns.boxplot(x='classifier',y='accuracy',data=df_toplot)
            if PERSONALIZED:
                plt.savefig("/Users/Matt/Desktop/classification_output/%s_%s_%s_personalized.png" % (p, dfname,featuregroupname))
            else:
                plt.savefig("/Users/Matt/Desktop/classification_output/%s_%s_%s.png" % (p, dfname,featuregroupname))
            plt.clf()
        # for (featuregroupname, featuregroup, clf) in featurenames_and_featuresets:
        #     for p in PREDICTIONS:
        #         df_toplot = prediction_results_dump[(prediction_results_dump['dfname'] == dfname) & (prediction_results_dump['featuregroupname'] == featuregroupname) & (prediction_results_dump['prediction'] == p)]
        #         print(df_toplot)
        #         print(df_toplot.columns.values)
        #         sns.boxplot(x='classifier',y='accuracy',data=df_toplot)
        #         plt.savefig("/Users/Matt/Desktop/classification_output/%s_%s_%s.png" % (p, dfname,featuregroupname))
        #         plt.clf()

    print("COMPARE SIGNIFICANCE")

    for (dfname1,prediction1,dfname2,prediction2) in [
        ('trec_firstquery', 'facet_goal', 'trec_firstquery', 'facet_goal'),
        ('trec_firstquery', 'facet_product', 'trec_firstquery', 'facet_product'),
        ('trec_firstquery', 'facet_trec2014tasktype', 'trec_firstquery',
         'facet_trec2014tasktype'),

        ('trec_wholesession', 'facet_goal', 'trec_wholesession', 'facet_goal'),
        ('trec_wholesession', 'facet_product', 'trec_wholesession', 'facet_product'),
        ('trec_wholesession', 'facet_trec2014tasktype', 'trec_wholesession',
         'facet_trec2014tasktype'),

        ('nsf_firstquery', 'facet_goal', 'nsf_firstquery', 'facet_goal'),
        ('nsf_firstquery', 'facet_product', 'nsf_firstquery', 'facet_product'),
        ('nsf_firstquery', 'facet_trec2014tasktype', 'nsf_firstquery',
         'facet_trec2014tasktype'),

        ('nsf_wholesession', 'facet_goal', 'nsf_wholesession', 'facet_goal'),
        ('nsf_wholesession', 'facet_product', 'nsf_wholesession', 'facet_product'),
        ('nsf_wholesession', 'facet_trec2014tasktype', 'nsf_wholesession',
         'facet_trec2014tasktype'),


        #Testing best in first vs best in whole
        ('trec_firstquery', 'facet_goal', 'trec_wholesession', 'facet_goal'),
        ('trec_firstquery', 'facet_product', 'trec_wholesession', 'facet_product'),
        ('trec_firstquery', 'facet_trec2014tasktype', 'trec_firstquery',
         'facet_trec2014tasktype'),



        ('nsf_firstquery', 'facet_goal', 'nsf_wholesession', 'facet_goal'),
        ('nsf_firstquery', 'facet_product', 'nsf_wholesession', 'facet_product'),
        ('nsf_firstquery', 'facet_trec2014tasktype', 'nsf_wholesession',
         'facet_trec2014tasktype'),]:

        # [("stratified", ['dummy1'], 'str'),
        #  ("mostfrequent_baseline", ['dummy2'], 'mfq'),
        #  ("allfeatures_ada", FEATURES, 'ada'),
        #  ("allfeatures_knn", FEATURES, 'knn'),
        #  ("allfeatures_gnb", FEATURES, 'gnb'),
        #  # ("allfeatures_ovr", FEATURES, 'ovr'),
        #  # ("allfeatures_ovo", FEATURES, 'ovo'),
        #  ("allfeatures_svc", FEATURES, 'svc'),
        #  ("allfeatures_rfc", FEATURES, 'rfc'),
        #  ("allfeatures_mlp", FEATURES, 'mlp'),
        #  ("allfeatures_dct", FEATURES, 'dct')
        #  ]


        for modelname1 in ['str','mfq','ada','knn','gnb','svc','rfc','mlp','dct']:
            for modelname2 in ['str','mfq','ada','knn','gnb','svc','rfc','mlp','dct']:
                stat = None
                p = None
                accuracy1 = prediction_accuracy_results[(dfname1,modelname1,prediction1)]
                accuracy2 = prediction_accuracy_results[(dfname2,modelname2,prediction2)]
                (stat1, pnorm1) = stats.normaltest(accuracy1)
                (stat2, pnorm2) = stats.normaltest(accuracy2)
                if (pnorm1 < .05 or pnorm2 < .05):
                    (stat, p) = stats.mannwhitneyu(accuracy1,accuracy2)
                else:
                    print("normal ttest")
                    (stat, p) = stats.ttest_ind(accuracy1, accuracy2)

                print((dfname1,modelname1,prediction1,dfname2,modelname2,prediction2))
                print(np.mean(accuracy1),"vs",np.mean(accuracy2))
                print("significance",p)

        # for (dfname1, modelname1, prediction1, dfname2, modelname2, prediction2) in [
        #
        #     # Testing best against baseline
        #     ('trec_firstquery', 'svc', 'facet_goal', 'trec_firstquery', 'str', 'facet_goal'),
        #     ('trec_firstquery', 'mlp', 'facet_product', 'trec_firstquery', 'mfq', 'facet_product'),
        #     ('trec_firstquery', 'knn', 'facet_trec2014tasktype', 'trec_firstquery', 'str',
        #      'facet_trec2014tasktype'),
        #
        #     ('trec_wholesession', 'gnb', 'facet_goal', 'trec_wholesession', 'str', 'facet_goal'),
        #     ('trec_wholesession', 'rfc', 'facet_product', 'trec_wholesession', 'mfq', 'facet_product'),
        #     ('trec_wholesession', 'rfc', 'facet_trec2014tasktype', 'trec_wholesession', 'str',
        #      'facet_trec2014tasktype'),
        #
        #     ('nsf_firstquery', 'svc', 'facet_goal', 'nsf_firstquery', 'mfq', 'facet_goal'),
        #     ('nsf_firstquery', 'mlp', 'facet_product', 'nsf_firstquery', 'str', 'facet_product'),
        #     ('nsf_firstquery', 'svc', 'facet_trec2014tasktype', 'nsf_firstquery', 'mfq',
        #      'facet_trec2014tasktype'),
        #
        #     ('nsf_wholesession', 'mlp', 'facet_goal', 'nsf_wholesession', 'mfq', 'facet_goal'),
        #     ('nsf_wholesession', 'knn', 'facet_product', 'nsf_wholesession', 'str', 'facet_product'),
        #     ('nsf_wholesession', 'knn', 'facet_trec2014tasktype', 'nsf_wholesession', 'mfq',
        #      'facet_trec2014tasktype'),
        #
        #     # Testing best in first vs best in whole
        #     ('trec_firstquery', 'svc', 'facet_goal', 'trec_wholesession', 'gnb', 'facet_goal'),
        #     ('trec_firstquery', 'mlp', 'facet_product', 'trec_wholesession', 'rfc', 'facet_product'),
        #     ('trec_firstquery', 'knn', 'facet_trec2014tasktype', 'trec_firstquery', 'rfc',
        #      'facet_trec2014tasktype'),
        #
        #     ('nsf_firstquery', 'mfq', 'facet_goal', 'nsf_wholesession', 'mfq', 'facet_goal'),
        #     ('nsf_firstquery', 'mlp', 'facet_product', 'nsf_wholesession', 'knn', 'facet_product'),
        #     ('nsf_firstquery', 'mfq', 'facet_trec2014tasktype', 'nsf_wholesession', 'mfq',
        #      'facet_trec2014tasktype'),
        #
        # ]:


        #                 if (pnorm1 < .05 or pnorm2 < .05):
        #                     (stat, p) = stats.mannwhitneyu(groups_featurevalues[0], groups_featurevalues[1])
        #                 else:
        #                     print("normal ttest")
        #                     (stat, p) = stats.ttest_ind(groups_featurevalues[0], groups_featurevalues[1])

    # prediction_results_all_ci = pd.concat(prediction_results_all_ci)
    # print(prediction_results_all_ci)
    # print("LEN",len(prediction_results_all_ci.index))
    # prediction_results_all = pd.DataFrame(prediction_results_all)
    #
    # prediction_results_q_ci = prediction_results_all.copy(deep=True)
    # prediction_results_a_ci = prediction_results_all.copy(deep=True)
    # prediction_results_q_ci = prediction_results_q_ci[prediction_results_q_ci['queries_num_session'].notna()]
    # prediction_results_a_ci = prediction_results_a_ci[prediction_results_a_ci['total_actions_count'].notna()]
    #
    # prediction_results_q = prediction_results_all.copy(deep=True)
    # prediction_results_a = prediction_results_all.copy(deep=True)
    # prediction_results_q = prediction_results_q[prediction_results_q['queries_num_session'].notna()]
    # prediction_results_a = prediction_results_a[prediction_results_a['total_actions_count'].notna()]
    #
    # print("QCI",len(prediction_results_q_ci.index))
    # print("ACI", len(prediction_results_a_ci.index))
    #
    # for p in set(prediction_results_q['prediction'].tolist()):
    #     df = prediction_results_q[prediction_results_q['prediction']==p]
    #     print("QNUM",set(prediction_results_q['queries_num_session'].tolist()))
    #     ax = sns.pointplot(x='queries_num_session', y='accuracy', hue='classifier', ci=None,data=df)
    #     fig = ax.get_figure()
    #     if LEQ:
    #         fig.savefig("/Users/Matt/Desktop/output/%s_q_accuracy_leq.png" % p)
    #     else:
    #         fig.savefig("/Users/Matt/Desktop/output/%s_q_accuracy.png"%p)
    #     plt.clf()
    #
    #     df = prediction_results_q_ci[prediction_results_q_ci['prediction'] == p]
    #     print(df)
    #     print("LEN", len(df.index))
    #     ax = sns.pointplot(x='queries_num_session', y='accuracy', hue='classifier', data=df)
    #     fig = ax.get_figure()
    #     if LEQ:
    #         fig.savefig("/Users/Matt/Desktop/output/%s_q_accuracy_ci_leq.png" % p)
    #     else:
    #         fig.savefig("/Users/Matt/Desktop/output/%s_q_accuracy_ci.png" % p)
    #     plt.clf()
    #
    # for p in set(prediction_results_a['prediction'].tolist()):
    #     df = prediction_results_a[prediction_results_a['prediction'] == p]
    #     print("ACT", set(prediction_results_a['total_actions_count'].tolist()))
    #     ax = sns.pointplot(x='total_actions_count', y='accuracy', hue='classifier', ci=None, data=df)
    #     fig = ax.get_figure()
    #     if LEQ:
    #         fig.savefig("/Users/Matt/Desktop/output/%s_a_accuracy_leq.png" % p)
    #     else:
    #         fig.savefig("/Users/Matt/Desktop/output/%s_a_accuracy.png" % p)
    #     plt.clf()
    #
    #     df = prediction_results_a_ci[prediction_results_a_ci['prediction'] == p]
    #     print(df)
    #     print("LEN", len(df.index))
    #     ax = sns.pointplot(x='total_actions_count', y='accuracy', hue='classifier', data=df)
    #     fig = ax.get_figure()
    #     if LEQ:
    #         fig.savefig("/Users/Matt/Desktop/output/%s_a_accuracy_ci_leq.png" % p)
    #     else:
    #         fig.savefig("/Users/Matt/Desktop/output/%s_a_accuracy_ci.png" % p)
    #     plt.clf()


























    # dumpnum = 10
    # RUN_CT = 0
    # ALL_FEATURES = FEATURES
    # lr_results = []
    # result_dump_dataframe = None
    # prediction_dataset = dataset_tail






    # tasks = []
    # for (n,group) in prediction_dataset.groupby('session_num'):
    #     tasks += [group['facet_trec2014tasktype'].tolist()[0]]




    # for (featuregroupname,featuregroup,clf) in featurenames_and_featuresets:
    #
    #     # LOGISTIC REGRESSION
    #     print(featuregroupname,featuregroup,clf)
    #     lr_res = run_prediction(prediction_dataset.copy(),input_features=featuregroup,clf=clf)
    #
    #     for r in lr_res:
    #         r['splitby'] = 'all'
    #         r['value'] = 'all'
    #         r['featureset'] = featuregroupname
    #         # print(r)
    #
    #         y_true,y_pred = r.pop('y_true'),r.pop('y_pred')
    #         y_true,y_pred = pd.Series(y_true),pd.Series(y_pred)
    #         # y_true,y_pred,y_score = r.pop('y_true'),r.pop('y_pred'),r.pop('y_score')
    #         # y_true,y_pred,y_score = pd.Series(y_true),pd.Series(y_pred),pd.Series(y_score)
    #         splitby_col = ['all' for v in y_true]
    #         value_col = [PREDICTION_VALUE_TO_ACRONYM['all'] for v in y_true]
    #         featureset_col = [FEATURESET_TO_ACRONYM[featuregroupname] for v in y_true]
    #         intention_col = [INTENTION_COLUMN_TO_ACRONYM[r['intention']] for v in y_true]
    #         runct_col = [r['run_ct'] for v in y_true]
    #
    #         result_temp_df = pd.DataFrame({'run':runct_col,'intention':intention_col,'featureset':featureset_col,
    #         'splitby':splitby_col,'value':value_col,
    #         'y_true':y_true,'y_pred':y_pred,
    #                                        # 'y_score':y_score
    #                                        })
    #         if result_dump_dataframe is None:
    #             result_dump_dataframe = result_temp_df
    #         else:
    #             result_dump_dataframe = pd.concat([result_dump_dataframe,result_temp_df])
    #     lr_results += [r for r in lr_res]
    #
    #
    #
    # result_dump_dataframe.to_csv('../../reports/prediction_lr_fulldump_%d.csv'%dumpnum)
    # lr_results = pd.DataFrame(lr_results)
    # lr_results.to_csv('../../reports/prediction_lr_scoreresults.csv')




    #Plot results
    # lr_results = pd.DataFrame.from_csv('../../reports/prediction_lr_scoreresults.csv')
    # lr_results = lr_results.fillna(value='')
    # lr_results['facet_names_values'] = lr_results['value'].map(lambda x: FACETVALUE_TO_SHORTNAME[x])
    #
    # lr_mean_results = []
    # for (name,group) in lr_results.groupby(['splitby','value','intention','featureset']):
    #
    #     # try:
    #     #     mean_ap = np.mean(group['ap'])
    #     # except TypeError:
    #     #     mean_ap = 0
    #     #
    #     #
    #     # try:
    #     #     mean_aucroc = np.mean(group['aucroc'])
    #     # except TypeError:
    #     #     mean_aucroc = 0
    #     #
    #     # try:
    #     #     mean_recall = np.mean(group['recall'])
    #     # except TypeError:
    #     #     mean_recall = 0
    #     #
    #     # try:
    #     #     mean_precision = np.mean(group['precision'])
    #     # except TypeError:
    #     #     mean_precision = 0
    #     #
    #     # try:
    #     #     mean_f1 = np.mean(group['f1'])
    #     # except TypeError:
    #     #     mean_f1 = 0
    #
    #     try:
    #         mean_accuracy = np.mean(group['accuracy'])
    #     except TypeError:
    #         mean_accuracy = 0
    #     lr_mean_results += [{'intention':name[2],'featureset':name[3],'splitby':name[0],'value':name[1],
    #     'accuracy':mean_accuracy}]



    # lr_mean_results = pd.DataFrame(lr_mean_results)
    #
    # personalized_string = ''
    # if PERSONALIZED:
    #     personalized_string = '_personalized'
    #
    # for (featuregroupname,featuregroup,clf) in featurenames_and_featuresets:
    #     for splitbyvalues in set(['all']):
    #         lr_mean_results_subframe = lr_mean_results[lr_mean_results['featureset']==featuregroupname]
    #         lr_mean_results_subframe = lr_mean_results[lr_mean_results['value']==splitbyvalues]
    #         print(featuregroupname,splitbyvalues)
    #         print("LEN",len(lr_mean_results_subframe.index))
    #         for scoretype in ['accuracy']:
    #         # for scoretype in ['accuracy','f1','precision','recall','aucroc','ap']:
    #             pivot_table = lr_mean_results_subframe.pivot(index='intention', columns='featureset', values=scoretype)
    #             pivot_table.to_csv("../../reports/lr_prediction_pivot_by%s_%s%s.csv"%(splitbyvalues,scoretype,personalized_string))
    #
    #
    # exit()
    # #####
    # #
    # # END PREDICTION RESULTS
    # #
    # #####








# INTENTION_COLUMN_TO_ACRONYM = {
#     'userID':'UID',
# 'facet_trec2014tasktype':'TASKTYPE',
# 'facet_product':'PRODUCT',
# 'facet_level':'LEVEL',
# 'facet_goal':'GOAL',
# 'facet_named':'NAMED',
# 'facet_complexity':"COMPLEX",
# 'facet_complexity_merged':"COMPLEX_MERGED",
# 'facet_complexity_create':"COMPLEX_CREATE",
# 'facet_goalplusproduct':"GOALPLUSPRODUCT",
#     'facet_trec2014knownitem':'KNOWNITEM',
#                               'facet_trec2014knownsubject':'KNOWNSUBJECT',
# 'facet_trec2014interpretive':'INTERPRETIVE',
# 'facet_trec2014exploratory':'EXPLORATORY'
# }
#
# FACETVALUE_TO_SHORTNAME = {
# '':'ALL',
# None:'ALL',
# 'Factual':'prod_FACT',
# 'Intellectual':'prod_INTEL',
# 'Document':'level_DOC',
# 'Segment':'level_SEG',
# 'Amorphous':'goal_AMOR',
# 'Specific':'goal_SPEC',
# 0:'named_NO',
# 1:'named_YES',
# '0':'named_NO',
# '1':'named_YES',
# 'Coelacanths':'topic_COE',
# 'Methane clathrates and global warming':'topic_MET',
# 'Copy Editing':'type_CPE',
# 'Interview Preparation':'type_INT',
# 'Story Pitch':'type_STP',
# 'Relationships':'type_REL',
# 'all':'type_all'
# }


# PREDICTION_VALUE_TO_ACRONYM = {
# 'Relationships':'REL',
# 1:'TRU',
# 'Story Pitch':'STP',
# 'Factual':'FAC',
# 'all':'ALL',
# 'Amorphous':'AMO',
# 'Intellectual':'INT',
# 'Specific':'SPE',
# 0:'FAL',
# 'Copy Editing':'CPE',
# 'Coelacanths':'COE',
# 'Interview Preparation':'INT',
# 'Document':'DOC',
# 'Segment':'SEG',
# 'Methane clathrates and global warming':'MTH'
# }