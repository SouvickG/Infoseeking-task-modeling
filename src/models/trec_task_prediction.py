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

PERSONALIZED = False

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

PREDICTION_VALUE_TO_ACRONYM = {
'Relationships':'REL',
1:'TRU',
'Story Pitch':'STP',
'Factual':'FAC',
'all':'ALL',
'Amorphous':'AMO',
'Intellectual':'INT',
'Specific':'SPE',
0:'FAL',
'Copy Editing':'CPE',
'Coelacanths':'COE',
'Interview Preparation':'INT',
'Document':'DOC',
'Segment':'SEG',
'Methane clathrates and global warming':'MTH'
}

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
'allfeatures_gsc':'ALL_gsc',
}



INTENTION_COLUMN_TO_ACRONYM = {
    'userID':'UID',
'facet_trec2014tasktype':'TASKTYPE',
'facet_product':'PRODUCT',
'facet_level':'LEVEL',
'facet_goal':'GOAL',
'facet_named':'NAMED',
'facet_complexity':"COMPLEX",
'facet_complexity_merged':"COMPLEX_MERGED",
'facet_complexity_create':"COMPLEX_CREATE",
'facet_goalplusproduct':"GOALPLUSPRODUCT",
    'facet_trec2014knownitem':'KNOWNITEM',
                              'facet_trec2014knownsubject':'KNOWNSUBJECT',
'facet_trec2014interpretive':'INTERPRETIVE',
'facet_trec2014exploratory':'EXPLORATORY'
}

FACETVALUE_TO_SHORTNAME = {
'':'ALL',
None:'ALL',
'Factual':'prod_FACT',
'Intellectual':'prod_INTEL',
'Document':'level_DOC',
'Segment':'level_SEG',
'Amorphous':'goal_AMOR',
'Specific':'goal_SPEC',
0:'named_NO',
1:'named_YES',
'0':'named_NO',
'1':'named_YES',
'Coelacanths':'topic_COE',
'Methane clathrates and global warming':'topic_MET',
'Copy Editing':'type_CPE',
'Interview Preparation':'type_INT',
'Story Pitch':'type_STP',
'Relationships':'type_REL',
'all':'type_all'
}

METADATA=[
    'facet_product','facet_goal','facet_trec2014tasktype',
    'userID',
    'Unnamed: 0','questionID','queryID','session_num','interaction_num','action_type','local_timestamp','data','start_time','start_action','total_actions']


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

def plot_embedding(X,y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def run_logisticregression(df,input_features=None,clf=None):
    global FEATURES
    global RUN_CT
    features = None

    if input_features is None or input_features==['dummy1'] or input_features==['dummy2']:
        features = FEATURES
        # print "FEATURES",FEATURES[92:]
    else:
        features = input_features



    le = preprocessing.LabelEncoder()


    print("input features",features)

    # scaler = Normalizer()
    scaler = StandardScaler()
    results = []


    df_copy = df.copy()
    userIDs = set(df_copy['userID'].tolist())

    if not PERSONALIZED:
        userIDs = [1]

    for facet in PREDICTIONS:
        print(facet)
        for i in range(NUM_ITERS):
            print(i)
            RUN_CT += 1
            res = {'intention':facet}

            # 0) Slice to only feature columns and  test column
            # 0a) Convert data to np array
            # 1) Random permutation



            y_tests_all = []
            y_preds_all = []
            y_scores_all = []
            for u in userIDs:
                data_okay = True
                res = {'intention': facet}
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




                            # if True:
                            # # if facet=='userID':
                            #     # BEGIN PCA
                            #
                            #     pca = PCA(n_components=3)
                            #     target_names = list(set(list(df[facet].as_matrix())))
                            #     X_r = pca.fit(X_tofit).transform(X_tofit)
                            #
                            #     # lda = LinearDiscriminantAnalysis(n_components=2)
                            #     # X_r2 = lda.fit(X_tofit, y_tofit).transform(X_tofit)
                            #
                            #     print('explained variance ratio (first two components): %s'% str(pca.explained_variance_ratio_))
                            #     fig = plt.figure()
                            #     colors = ['navy', 'turquoise', 'darkorange', 'magenta', 'red','yellow']
                            #     lw = 2
                            #     ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
                            #
                            #     for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5], target_names):
                            #         print(y_tofit)
                            #         print(X_r[y_tofit == i, 0])
                            #         print(X_r.shape)
                            #         print(color,i,target_name)
                            #
                            #         ax.scatter(X_r[y_tofit == i, 0], X_r[y_tofit == i, 1], X_r[y_tofit == i, 2], color=color, alpha=.8, lw=lw,label=target_name)
                            #     plt.legend(loc='best', shadow=False, scatterpoints=1)
                            #     plt.title('PCA ')
                            #     plt.savefig('../../reports/%s_PCA.png'%facet)
                            #
                            #     # plt.figure()
                            #     # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
                            #     #     print(X_r2.shape)
                            #     #     plt.scatter(X_r2[y_tofit == i, 0], X_r2[y_tofit == i, 1], alpha=.8, color=color,label=target_name)
                            #     # plt.legend(loc='best', shadow=False, scatterpoints=1)
                            #     # plt.title('LDA of IRIS dataset')
                            #     # plt.savefig('query_lengths_bytopic.png')
                            #     # END PCA
                            #
                            #     # BEGIN Isomap
                            #     imap = Isomap(n_components=2)
                            #
                            #
                            #     target_names = list(set(list(df[facet].as_matrix())))
                            #     X_r = imap.fit(X_tofit).transform(X_tofit)
                            #     # exit()
                            #
                            #
                            #     plot_embedding(X_r,y_tofit,"Isomap projection")
                            #
                            #     plt.figure()
                            #     colors = ['navy', 'turquoise', 'darkorange', 'magenta', 'red', 'yellow']
                            #     lw = 2
                            #     for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5], target_names):
                            #         print(y_tofit)
                            #         print(X_r[y_tofit == i, 0])
                            #         print(X_r.shape)
                            #         print(color,i,target_name)
                            #
                            #         plt.scatter(X_r[y_tofit == i, 0], X_r[y_tofit == i, 1], color=color, alpha=.8, lw=lw,label=target_name)
                            #     plt.legend(loc='best', shadow=False, scatterpoints=1)
                            #     plt.title('Isomap')
                            #     plt.savefig('../../reports/%s_isomap.png' % facet)
                            #     # END Isomap



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
                                     }
                            if clf is not None:
                                # lr_model = clf_map[clf]
                                lr_model = Pipeline([('anova', anovakbest_filter), ('clf', clf_map[clf])])
                                # lr_model = Pipeline([('pca', pca), ('clf', clf_map[clf])])
                                # lr_model = Pipeline([('feature_selection', rfe), ('clf', clf_map[clf])])
                                # lr_model = Pipeline([('feature_selection', sfm), ('clf', clf_map[clf])])
                            else:
                                lr_model = Pipeline([('anova', anovakbest_filter), ('clf', gnb)])


                        scaler.fit(X_train)
                        lr_model.fit(scaler.transform(X_train),y_train)
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
    return results



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

if __name__=='__main__':

    dataset = pd.read_csv('../../data/interim/trec2014_task_prediction_features.csv')
    PREDICTIONS += [
        # 'facet_trec2014knownitem',
                    # 'facet_trec2014knownsubject',
                    # 'facet_trec2014interpretive',
                    # 'facet_trec2014exploratory'
                    ]

    FEATURES = [c for c in list(dataset.columns.values) if not c in PREDICTIONS + METADATA]

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


    # dataset['facet_trec2014knownitem'] = dataset['facet_trec2014tasktype'].apply(lambda x: x if x =='known-item' else 'not-'+'known-item')
    # dataset['facet_trec2014knownsubject'] = dataset['facet_trec2014tasktype'].apply(lambda x: x if x =='known-subject' else 'not-'+'known-subject')
    # dataset['facet_trec2014interpretive'] = dataset['facet_trec2014tasktype'].apply(lambda x: x if x =='interpretive' else 'not-'+'interpretive')
    # dataset['facet_trec2014exploratory'] = dataset['facet_trec2014tasktype'].apply(lambda x: x if x =='exploratory' else 'not-'+'exploratory')

    dataset_tail = []
    dataset_endsteponly = dataset[dataset['start_action']==False]
    for (n, group) in dataset_endsteponly.groupby(['session_num']):
        dataset_tail += [group.tail(1)]
    dataset_tail = pd.concat(dataset_tail)

    sessions_per_user_df = pd.read_csv('../../data/interim/trec2014_sessions_per_user_nonnegativeuserid.csv')
    active_users = list(set(sessions_per_user_df[sessions_per_user_df['num_sessions'] >=40]['user_num'].tolist()))
    print("N USERS",len(list(set(active_users))))


    users_per_topic_df = pd.read_csv('../../data/interim/trec2014_users_per_topic_nonnegativeuserid.csv')
    active_topics = list(set(users_per_topic_df[users_per_topic_df['num_sessions'] >= 20]['topic_num'].tolist()))

    sessions_overview_df = pd.read_csv('../../data/interim/trec2014_sessions_overview.csv')
    # short_sessions = list(set(sessions_overview_df[sessions_overview_df['queries_num_session'] <= 3]['session_num'].tolist()))



    dumpnum = 10
    RUN_CT = 0
    NUM_ITERS = 60
    ALL_FEATURES = FEATURES

    featurenames_and_featuresets = [("stratified", ['dummy1'],None), ("mostfrequent_baseline", ['dummy2'],None),
                                  # ("allfeatures_default", ALL_FEATURES,None),
                                  #   ("allfeatures_svc", ALL_FEATURES, 'svc'),
                                  #   ("allfeatures_gsc", ALL_FEATURES, 'gsc'),
                                    ("allfeatures_ada", ALL_FEATURES, 'ada'),
                                    ("allfeatures_knn", ALL_FEATURES, 'knn'),
                                    ("allfeatures_gnb", ALL_FEATURES, 'gnb'),
                                    ("allfeatures_ovr", ALL_FEATURES, 'ovr'),
                                    ("allfeatures_ovo", ALL_FEATURES, 'ovo'),
                                    ("allfeatures_mlp", ALL_FEATURES, 'mlp')
                                    ]
    # featurenames_and_featuresets = [("allfeatures", ALL_FEATURES)]

    lr_results = []
    result_dump_dataframe = None





    prediction_dataset = dataset_tail
    # prediction_dataset = dataset_tail[dataset_tail['userID'].isin(active_users)]



    # prediction_dataset = dataset_tail[~dataset_tail['session_num'].isin(short_sessions)]
    # prediction_dataset = dataset[dataset['userID'].isin(active_users)]


    # # BEGIN TRIALS
    #
    # N_TOTAL_TRIALS = 10000
    # n_conducted_trials = 0
    #
    # trials_df = []
    # n_significant = dict()
    # for feature in FEATURES:
    #     for p in [.05, .01, .001]:
    #         n_significant[('facet_goal',feature,p)] = 0
    #         n_significant[('facet_product', feature, p)] = 0
    #         n_significant[('facet_trec2014tasktype', feature, p)] = 0
    #
    #
    # task_types = list(set(dataset_tail['facet_trec2014tasktype'].tolist()))
    #
    # users_onetask = []
    # task_type_count = None
    # tasks_assigned = []
    # session_nums = []
    #
    # for (userID, group) in prediction_dataset.groupby(['userID']):
    #     task_type_count = Counter(group['facet_trec2014tasktype'].tolist())
    #     n_sessions = len(list(set(group['session_num'].tolist())))
    #     if len(task_type_count)==1:
    #         users_onetask+=[userID]
    #         tasks_assigned += [group['facet_trec2014tasktype'].tolist()[0]]
    #         session_nums += [group['session_num'].tolist()[0]]
    #
    #
    # n_left_counter = Counter(tasks_assigned)
    # for n in n_left_counter.keys():
    #     n_left_counter[n] = max(0, 65 - n_left_counter[n])
    #
    #
    # # 194 chosen at this point, 66 are random
    # print(session_nums)
    # print(len(session_nums))
    # print(len(list(set(session_nums))))
    # print(n_left_counter)
    # # exit()
    #
    #
    # prediction_dataset_multipletasks = prediction_dataset[~prediction_dataset['userID'].isin(users_onetask)]
    #
    # while(n_conducted_trials < N_TOTAL_TRIALS):
    #     print("N CONDUCTED",n_conducted_trials)
    #
    #     tasks_multipletasksusers = []
    #     sessions_multipletaskusers = []
    #     users_multipletasks_assigned = []
    #
    #     valid = True
    #
    #
    #     data_usersleft = prediction_dataset_multipletasks[~prediction_dataset_multipletasks['userID'].isin(users_multipletasks_assigned)]
    #
    #     for (tasktype,count) in n_left_counter.most_common(4):
    #         for _ in range(count):
    #             session_nums_multipletaskusers = data_usersleft[data_usersleft['facet_trec2014tasktype']==tasktype]['session_num'].tolist()
    #             if len(session_nums_multipletaskusers)==0:
    #                 valid = False
    #                 break
    #             random_session = random.choice(session_nums_multipletaskusers)
    #
    #             assigned_user = data_usersleft[data_usersleft['session_num']==random_session]['userID'].tolist()[0]
    #             sessions_multipletaskusers += [random_session]
    #             users_multipletasks_assigned += [assigned_user]
    #             tasks_multipletasksusers += [tasktype]
    #             data_usersleft = prediction_dataset_multipletasks[
    #                 ~prediction_dataset_multipletasks['userID'].isin(users_multipletasks_assigned)]
    #         if valid == False:
    #             break
    #
    #
    #     assert len(users_multipletasks_assigned) == len(set(users_multipletasks_assigned))
    #     total_counter = Counter(tasks_multipletasksusers)+Counter(tasks_assigned)
    #     if not valid:
    #         if max(total_counter.values())-min(total_counter.values())>3:
    #             # print("FAILURE!",Counter(tasks_multipletasksusers) + Counter(tasks_assigned))
    #             continue
    #         # else:
    #         #     print(users_multipletasks_assigned)
    #
    #     # print("success!",print(Counter(tasks_multipletasksusers)+Counter(tasks_assigned)))
    #     # print("success!",print(Counter(tasks_multipletasksusers)))
    #
    #     assert len(sessions_multipletaskusers) + len(users_onetask) in [260,259,258,257,256]
    #     n_conducted_trials += 1
    #
    #     # statistical_significance_dataset = [prediction_dataset[prediction_dataset['userID'].isin(users_onetask)]]
    #     # statistical_significance_dataset += [prediction_dataset[prediction_dataset['session_num'].isin(sessions_multipletaskusers+session_nums)]]
    #     # statistical_significance_dataset = pd.concat(statistical_significance_dataset)
    #
    #     # print(sessions_multipletaskusers + session_nums)
    #     statistical_significance_dataset = prediction_dataset[prediction_dataset['session_num'].isin(sessions_multipletaskusers + session_nums)]
    #     assert len(sessions_multipletaskusers) + len(users_onetask) == len(statistical_significance_dataset.index)
    #     # print(statistical_significance_dataset)
    #
    #     for target in ['facet_trec2014tasktype', 'facet_goal', 'facet_product']:
    #         groups = []
    #         for (n, group) in statistical_significance_dataset.groupby(target):
    #             groups += [group]
    #         for feature in FEATURES:
    #             groups_featurevalues = [g[feature] for g in groups]
    #             (stat,p) = (None,None)
    #             if target =='facet_trec2014tasktype':
    #                 (stat1,pnorm1) = stats.normaltest(groups_featurevalues[0])
    #                 (stat2, pnorm2) = stats.normaltest(groups_featurevalues[1])
    #                 (stat3, pnorm3) = stats.normaltest(groups_featurevalues[2])
    #                 (stat4, pnorm4) = stats.normaltest(groups_featurevalues[3])
    #                 if(pnorm1 < .05 or pnorm2 < .05 or pnorm3 < .05 or pnorm4 < .05):
    #                     (stat,p) = stats.kruskal(groups_featurevalues[0],groups_featurevalues[1],groups_featurevalues[2],groups_featurevalues[3])
    #                 else:
    #                     print("normal ANOVA")
    #                     (stat, p) = stats.f_oneway(groups_featurevalues[0], groups_featurevalues[1],
    #                                               groups_featurevalues[2], groups_featurevalues[3])
    #             else:
    #                 (stat1, pnorm1) = stats.normaltest(groups_featurevalues[0])
    #                 (stat2, pnorm2) = stats.normaltest(groups_featurevalues[1])
    #                 if (pnorm1 < .05 or pnorm2 < .05):
    #                     (stat, p) = stats.mannwhitneyu(groups_featurevalues[0], groups_featurevalues[1])
    #                 else:
    #                     print("normal ttest")
    #                     (stat, p) = stats.ttest_ind(groups_featurevalues[0], groups_featurevalues[1])
    #
    #             trials_df += [{'n_trial': n_conducted_trials, 'target': target, 'feature': feature, 'pvalue': p}]
    #             for pthreshold in [.05,.01,.001]:
    #                 n_significant[(target,feature,pthreshold)] += pthreshold > p
    #
    #
    #
    # df = []
    # for (i,j,k) in n_significant.keys():
    #     df += [{'target':i,'feature':j,'p':k,'count':n_significant[(i,j,k)],'probability':n_significant[(i,j,k)]/float(N_TOTAL_TRIALS)}]
    #
    # print(pd.DataFrame(df).sort_values(['target','count'],ascending=[0,0]))
    # pd.DataFrame(df).sort_values(['target', 'probability'], ascending=[0, 0]).to_csv('/Users/Matt/Desktop/significance_trials.csv')
    # pd.DataFrame(trials_df).to_csv('/Users/Matt/Desktop/trials_dump.csv')
    # exit()
    #
    # while n_conducted_trials < N_TOTAL_TRIALS:
    #     print("N CONDUCTED",n_conducted_trials)
    #     session_nums = []
    #     session_nums_onesession = []
    #     tasktype_count = None
    #     for (n,group) in prediction_dataset.groupby(['userID']):
    #         task_type_count = Counter(prediction_dataset[prediction_dataset['session_num'].isin(session_nums)]['facet_trec2014tasktype'].tolist())
    #         user_session_nums = list(set(group['session_num'].tolist()))
    #         task_types_present = []
    #         task_types_randomchoice = []
    #         for(n2,group2) in group.groupby(['facet_trec2014tasktype']):
    #             count = task_type_count[n2]
    #             weight = (260-count)/260
    #             task_types_randomchoice += [n2]*int(weight*260)
    #
    #         print(Counter(task_types_randomchoice))
    #         usertasktype = random.choice(task_types_randomchoice)
    #         usersessions = group[group['facet_trec2014tasktype']==usertasktype]['session_num'].tolist()
    #         session_nums += [random.choice(usersessions)]
    #
    #
    #
    #         # session_nums += [random.choice(user_session_nums)]
    #
    #         # if len(group.index)==1:
    #         #     session_nums_onesession += [session_nums[-1]]
    #
    #
    #     # print(Counter(sessions_onesession['facet_trec2014tasktype'].tolist()))
    #     # exit()
    #     # print(session_nums)
    #     sessions = prediction_dataset[prediction_dataset['session_num'].isin(session_nums)]
    #     valid = True
    #     task_type_count = {}
    #     for task_type in task_types:
    #         n_sessions = len(sessions[sessions['facet_trec2014tasktype']==task_type].index)
    #         task_type_count[task_type] = n_sessions
    #
    #     if max(task_type_count.values())-min(task_type_count.values()) >= 2:
    #         print(max(task_type_count.values())-min(task_type_count.values()))
    #         continue
    #
    #     n_conducted_trials += 1
    #
    # exit()
    # # END TRIALS


    # print("RES",Counter(prediction_dataset['facet_trec2014tasktype']))
    # exit()
    tasks = []
    for (n,group) in prediction_dataset.groupby('session_num'):
        tasks += [group['facet_trec2014tasktype'].tolist()[0]]


    # ki_data = prediction_dataset[prediction_dataset['facet_trec2014tasktype']=='known-item']
    # ks_data = prediction_dataset[prediction_dataset['facet_trec2014tasktype']=='known-subject']
    # interp_data = prediction_dataset[prediction_dataset['facet_trec2014tasktype']=='interpretive']
    # explor_data = prediction_dataset[prediction_dataset['facet_trec2014tasktype']=='exploratory']
    # spec_data = prediction_dataset[prediction_dataset['facet_goal']=='specific']
    # amor_data = prediction_dataset[prediction_dataset['facet_goal'] == 'amorphous']
    # fact_data = prediction_dataset[prediction_dataset['facet_product'] == 'factual']
    # intel_data = prediction_dataset[prediction_dataset['facet_product'] == 'intellectual']
    #
    # # print(len(spec_data.index),len(amor_data.index),len(fact_data.index),len(intel_data.index))
    #
    # user_data = []
    # for u in list(set(prediction_dataset['userID'].tolist())):
    #     user_data += [prediction_dataset[prediction_dataset['userID']==u]]
    #
    # n_features = len(FEATURES)
    # n_users = len(user_data)
    # print("TOTAL FEATURES",n_features)
    # print("TOTAL USERS",n_users)
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
    #     if p < pval:
    #         print("GOALFEATURE",attribute,p)
    #     (h, p) = stats.mannwhitneyu(fact_data[attribute],intel_data[attribute])
    #     n_prod_sig += p < pval
    #     if p < pval:
    #         print("PRODFEATURE",attribute,p)
    #
    #
    #     (h,p) = stats.kruskal(*[u[attribute] for u in user_data])
    #     n_user_sig += p < pval
    #
    #     # print(attribute+"-TASK",stats.kruskal(ki_data[attribute], ks_data[attribute],interp_data[attribute], explor_data[attribute]),.05/scipy.misc.comb(4,2))
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
    # # for attribute in ['query_lengths_mean','query_lengths_var','dwelltimes_mean_content_session',
    # #                   'dwelltimes_var_content_session','dwelltimes_mean_serp_session','dwelltimes_var_serp_session']:
    #     print(attribute+"-TASK",stats.kruskal(ki_data[attribute], ks_data[attribute],
    #                         interp_data[attribute], explor_data[attribute]),.05/scipy.misc.comb(4,2))
    #     print(attribute+"-USER",stats.kruskal(*[u[attribute] for u in user_data]),.05/scipy.misc.comb(len(user_data),2))

    # print(stats.kruskal(ki_data['dwelltimes_mean_content_session'],ks_data['dwelltimes_mean_content_session'],interp_data['dwelltimes_mean_content_session'],explor_data['dwelltimes_mean_content_session']))
    # print(stats.kruskal(ki_data['dwelltimes_var_content_session'],ks_data['dwelltimes_var_content_session'],interp_data['dwelltimes_var_content_session'],explor_data['dwelltimes_var_content_session']))
    #
    # print(stats.kruskal(ki_data['dwelltimes_mean_serp_session'], ks_data['dwelltimes_mean_serp_session'],
    #                     interp_data['dwelltimes_mean_serp_session'], explor_data['dwelltimes_mean_serp_session']))
    # print(stats.kruskal(ki_data['dwelltimes_var_serp_session'], ks_data['dwelltimes_var_serp_session'],
    #                     interp_data['dwelltimes_var_serp_session'], explor_data['dwelltimes_var_serp_session']))
    # exit()




    # prediction_dataset = prediction_dataset[prediction_dataset['userID']==18]
    # for f in FEATURES:
    #     for p in PREDICTIONS:
    #         print(f,p)
    #         ax = sns.boxplot(x=p, y=f, data=prediction_dataset)
    #         fig = ax.get_figure()
    #         fig.savefig('/Users/Matt/Desktop/plots/%s_%s.png'%(p,f))
    #         plt.clf()
    # exit()


    # sns_plot = sns.pairplot(prediction_dataset[FEATURES])
    # sns_plot.savefig("/Users/Matt/Desktop/plots/pairplots.png")
    # plt.clf()
    # exit()


    for (featuregroupname,featuregroup,clf) in featurenames_and_featuresets:

        # LOGISTIC REGRESSION
        print(featuregroupname,featuregroup,clf)
        lr_res = run_logisticregression(prediction_dataset.copy(),input_features=featuregroup,clf=clf)

        for r in lr_res:
            r['splitby'] = 'all'
            r['value'] = 'all'
            r['featureset'] = featuregroupname
            # print(r)

            y_true,y_pred = r.pop('y_true'),r.pop('y_pred')
            y_true,y_pred = pd.Series(y_true),pd.Series(y_pred)
            # y_true,y_pred,y_score = r.pop('y_true'),r.pop('y_pred'),r.pop('y_score')
            # y_true,y_pred,y_score = pd.Series(y_true),pd.Series(y_pred),pd.Series(y_score)
            splitby_col = ['all' for v in y_true]
            value_col = [PREDICTION_VALUE_TO_ACRONYM['all'] for v in y_true]
            featureset_col = [FEATURESET_TO_ACRONYM[featuregroupname] for v in y_true]
            intention_col = [INTENTION_COLUMN_TO_ACRONYM[r['intention']] for v in y_true]
            runct_col = [r['run_ct'] for v in y_true]

            result_temp_df = pd.DataFrame({'run':runct_col,'intention':intention_col,'featureset':featureset_col,
            'splitby':splitby_col,'value':value_col,
            'y_true':y_true,'y_pred':y_pred,
                                           # 'y_score':y_score
                                           })
            if result_dump_dataframe is None:
                result_dump_dataframe = result_temp_df
            else:
                result_dump_dataframe = pd.concat([result_dump_dataframe,result_temp_df])
        lr_results += [r for r in lr_res]



    result_dump_dataframe.to_csv('../../reports/prediction_lr_fulldump_%d.csv'%dumpnum)
    lr_results = pd.DataFrame(lr_results)
    lr_results.to_csv('../../reports/prediction_lr_scoreresults.csv')




    #Plot results
    lr_results = pd.DataFrame.from_csv('../../reports/prediction_lr_scoreresults.csv')
    lr_results = lr_results.fillna(value='')
    lr_results['facet_names_values'] = lr_results['value'].map(lambda x: FACETVALUE_TO_SHORTNAME[x])

    lr_mean_results = []
    for (name,group) in lr_results.groupby(['splitby','value','intention','featureset']):

        # try:
        #     mean_ap = np.mean(group['ap'])
        # except TypeError:
        #     mean_ap = 0
        #
        #
        # try:
        #     mean_aucroc = np.mean(group['aucroc'])
        # except TypeError:
        #     mean_aucroc = 0
        #
        # try:
        #     mean_recall = np.mean(group['recall'])
        # except TypeError:
        #     mean_recall = 0
        #
        # try:
        #     mean_precision = np.mean(group['precision'])
        # except TypeError:
        #     mean_precision = 0
        #
        # try:
        #     mean_f1 = np.mean(group['f1'])
        # except TypeError:
        #     mean_f1 = 0

        try:
            mean_accuracy = np.mean(group['accuracy'])
        except TypeError:
            mean_accuracy = 0







        lr_mean_results += [{'intention':name[2],'featureset':name[3],'splitby':name[0],'value':name[1],
        'accuracy':mean_accuracy}]

        # lr_mean_results += [{'intention':name[2],'featureset':name[3],'splitby':name[0],'value':name[1],
        # 'f1':mean_f1,'precision':mean_precision,'accuracy':mean_accuracy,
        #  'recall':mean_recall,'aucroc':mean_aucroc,'ap':mean_ap}]

        # lr_mean_results += [{'intention':name[2],'featureset':name[3],'splitby':name[0],'value':name[1],
        # 'f1':np.mean(group['f1']),'precision':np.mean(group['precision']),'accuracy':np.mean(group['accuracy']),
        #  'recall':np.mean(group['recall']),'aucroc':np.mean(group['aucroc']),'ap':np.mean(group['ap'])}]


    lr_mean_results = pd.DataFrame(lr_mean_results)

    personalized_string = ''
    if PERSONALIZED:
        personalized_string = '_personalized'

    for (featuregroupname,featuregroup,clf) in featurenames_and_featuresets:
        for splitbyvalues in set(['all']):
            lr_mean_results_subframe = lr_mean_results[lr_mean_results['featureset']==featuregroupname]
            lr_mean_results_subframe = lr_mean_results[lr_mean_results['value']==splitbyvalues]
            print(featuregroupname,splitbyvalues)
            print("LEN",len(lr_mean_results_subframe.index))
            for scoretype in ['accuracy']:
            # for scoretype in ['accuracy','f1','precision','recall','aucroc','ap']:
                pivot_table = lr_mean_results_subframe.pivot(index='intention', columns='featureset', values=scoretype)
                pivot_table.to_csv("../../reports/lr_prediction_pivot_by%s_%s%s.csv"%(splitbyvalues,scoretype,personalized_string))


    exit()
    # #####
    # #
    # # END PREDICTION RESULTS
    # #
    # #####
    #

    # BEGIN TODO 1
    # corr_results = []
    # lr_results = []
    #
    # THE_DATA_FRAME = full_features_queryend
    # full_features_queryend = full_features_queryend.drop('queryID',1)
    #
    # c = 0
    # for col_split in FACETS + TASK_COLUMN + TOPIC_COLUMN + ['all']:
    #     c += 1
    #     print "COL SPLIT",c,len(FACETS + TASK_COLUMN + TOPIC_COLUMN + ['all'])
    #     full_features_df_copy = THE_DATA_FRAME.copy()
    #     if col_split != 'all':
    #         for (n,group) in full_features_df_copy.groupby([col_split]):
    #             pass
    #             # CORRELATIONS
    #             # corr_subresults = run_correlation(group)
    #             # for r in corr_subresults:
    #             #     r['splitby'] = col_split
    #             #     r['value'] = n
    #             # corr_results += [r for r in corr_subresults]
    #
    #             # LOGISTIC REGRESSION
    #             # lr_res = run_logisticregression(group)
    #             # for r in lr_res:
    #             #     r['splitby'] = col_split
    #             #     r['value'] = n
    #             # lr_results += [r for r in lr_res]
    #             # print pd.DataFrame(lr_results)
    #
    #
    #
    #
    #
    # # # CORRELATIONS
    # # corr_subresults = run_correlation(THE_DATA_FRAME.copy())
    # # for r in corr_subresults:
    # #     r['splitby'] = 'all'
    # #     r['value'] = None
    # # corr_results += [r for r in corr_subresults]
    # # pd.DataFrame(corr_results).to_csv('/Users/Matt/Desktop/correlations_results.csv')
    # #
    # # # LOGISTIC REGRESSION
    # # lr_res = run_logisticregression(THE_DATA_FRAME.copy())
    # # for r in lr_res:
    # #     r['splitby'] = col_split
    # #     r['value'] = n
    # # lr_results += [r for r in lr_res]
    # # lr_results = pd.DataFrame(lr_results)
    # # lr_results.to_csv('/Users/Matt/Desktop/lr_results.csv')
    #
    #
    #
    #
    #
    # #
    # # Plotting
    # #
    #
    #
    # #Output CSVs
    # # for f in list(set(corr_results['facet_names_values'].tolist())):
    # #     corr_results[corr_results['facet_names_values']==f].to_csv(OUTFOLDER+"correlations/csv/correlations_taskfacet_%s.csv"%f)
    # #
    # # for i in INTENTION_COLUMNS:
    # #     corr_results[corr_results['intention']==i].to_csv(OUTFOLDER+"correlations/csv/correlations_intent_%s.csv"%i)
    #
    # corr_results = pd.DataFrame.from_csv('/Users/Matt/Desktop/correlations_results.csv')
    # corr_results = corr_results.fillna(value='')
    # print corr_results
    # lr_results = pd.DataFrame.from_csv('/Users/Matt/Desktop/lr_results.csv')
    # lr_results = lr_results.fillna(value='')
    # lr_results['facet_names_values'] = lr_results['value'].map(lambda x: facetvalue_to_columnname[x])
    # for f in list(set(lr_results['facet_names_values'].tolist())):
    #     lr_results[lr_results['facet_names_values']==f].to_csv(OUTFOLDER+"predictions/csv/predictions_taskfacet_%s.csv"%f)
    #
    # for i in INTENTION_COLUMNS:
    #     lr_results[lr_results['intention']==i].to_csv(OUTFOLDER+"predictions/csv/predictions_intent_%s.csv"%i)
    #
    #
    # # Output images
    # corr_results = pd.DataFrame(corr_results)
    # corr_results['facet_names_values'] = corr_results['value'].map(lambda x: facetvalue_to_columnname[x])
    #
    # facet_name_value_list = list(set(corr_results['facet_names_values'].tolist()))
    # #For each intention, how much do correlations differ per slice?
    # sns.set()
    # print "Significant Feature Similarity By Task"
    # for i in INTENTION_COLUMNS:
    #     correlation_byintention = corr_results[corr_results['intention'] == i]
    #     intention_plot_df = []
    #     print i
    #     for (x,y) in itertools.product(facet_name_value_list,facet_name_value_list):
    #
    #         facet1_frame =  correlation_byintention[correlation_byintention['facet_names_values'] == x]
    #         facet1_features = list(set(facet1_frame[facet1_frame['p']< P_SIGNIFICANT]['feature'].tolist()))
    #
    #
    #         facet2_frame =  correlation_byintention[correlation_byintention['facet_names_values'] == y]
    #         facet2_features = list(set(facet2_frame[facet2_frame['p']< P_SIGNIFICANT]['feature'].tolist()))
    #         intention_plot_df += [{'facetvalue1':x,'facetvalue2':y,'similarity':jaccard(facet1_features,facet2_features)}]
    #
    #
    #     intention_plot_df = pd.DataFrame(intention_plot_df)
    #     intention_plot_df = intention_plot_df.pivot("facetvalue1","facetvalue2","similarity")
    #
    #     plt.figure()
    #     # cmap=sns.palplot(sns.color_palette("RdBu_r", 7))
    #     sns_plot = sns.heatmap(intention_plot_df, annot=True,linewidths=.5)
    #     for item in sns_plot.get_yticklabels():
    #         item.set_rotation(0)
    #     for item in sns_plot.get_xticklabels():
    #         item.set_rotation(90)
    #
    #     plt.title('Significant Feature Similarity For Intention %s'%i)
    #     plt.tight_layout()
    #     plt.savefig(OUTFOLDER+"correlations/images/tasktopicsimilarity_intent_%s.png"%i)
    #
    #
    #
    #
    # print "Significant Feature Similarity By Intention"
    # for f in list(set(corr_results['facet_names_values'].tolist())):
    #     correlation_byfacet = corr_results[corr_results['facet_names_values'] == f]
    #     facet_plot_df = []
    #     print f
    #     for (x,y) in itertools.product(INTENTION_COLUMNS,INTENTION_COLUMNS):
    #
    #         facet1_frame =  correlation_byfacet[correlation_byfacet['intention'] == x]
    #         facet1_features = list(set(facet1_frame[facet1_frame['p']< P_SIGNIFICANT]['feature'].tolist()))
    #
    #         facet2_frame =  correlation_byfacet[correlation_byfacet['intention'] == y]
    #         facet2_features = list(set(facet2_frame[facet2_frame['p']< P_SIGNIFICANT]['feature'].tolist()))
    #
    #
    #         facet_plot_df += [{'intention1':x,'intention2':y,'similarity':jaccard(facet1_features,facet2_features)}]
    #
    #
    #     facet_plot_df = pd.DataFrame(facet_plot_df)
    #     facet_plot_df = facet_plot_df.pivot("intention1","intention2","similarity")
    #
    #     plt.figure()
    #     # cmap=sns.palplot(sns.color_palette("RdBu_r", 7))
    #     sns_plot = sns.heatmap(facet_plot_df, annot=True,linewidths=.5)
    #     for item in sns_plot.get_yticklabels():
    #         item.set_rotation(0)
    #     for item in sns_plot.get_xticklabels():
    #         item.set_rotation(90)
    #
    #     plt.title('Significant Feature Similarity for Facet %s'%f)
    #     plt.tight_layout()
    #     plt.savefig(OUTFOLDER+"correlations/images/tasktopicsimilarity_taskfacet_%s.png"%f)
    #
    #
    #
    #
    #
    #
    # # LOGISTIC REGRESSION
    #
    # lr_results = pd.DataFrame(lr_results)
    # lr_results['facet_names_values'] = lr_results['value'].map(lambda x: facetvalue_to_columnname[x])
    # lr_results_expanded = []
    # FEATURES_PLUS_WEIGHT = ['weight_'+ f for f in FEATURES]
    # for (n,row) in lr_results.iterrows():
    #     r = {}
    #     for column in lr_results:
    #         if not column in FEATURES_PLUS_WEIGHT:
    #             r[column] = row[column]
    #     for column in lr_results:
    #         if column in FEATURES_PLUS_WEIGHT:
    #             # print column,column in FEATURES_PLUS_WEIGHT
    #             # print "FEATURES",FEATURES_PLUS_WEIGHT
    #             r_copy = r.copy()
    #             r_copy['feature'] = column
    #             r_copy['weight'] = row[column]
    #             lr_results_expanded += [r_copy]
    #
    # lr_results_expanded = pd.DataFrame(lr_results_expanded)
    # print lr_results_expanded
    #
    #
    # # Classification accuracies
    # for f in list(set(lr_results['facet_names_values'].tolist())):
    #     prediction_byfacet = lr_results[lr_results['facet_names_values'] == f]
    #     print f
    #     for score in ["f1","accuracy"]:
    #         plt.figure()
    #         sns_plot = sns.boxplot(x="intention",y=score,data=prediction_byfacet)
    #         plt.title('Prediction Results (%s) By Facet - %s'%(score,f))
    #         for item in sns_plot.get_xticklabels():
    #             item.set_rotation(90)
    #         plt.ylim(ymax = 1, ymin = 0)
    #         plt.tight_layout()
    #         plt.savefig(OUTFOLDER+"predictions/images/prediction_%s_taskfacet_%s.png"%(score,f))
    #
    #
    #
    #
    #
    # # For each intention,facet pair, compute the ordering of the intentions
    # ORDERINGS = dict()
    # for f in list(set(lr_results['facet_names_values'].tolist())):
    #     print f
    #     for intention in INTENTION_COLUMNS:
    #         lr_subdf = lr_results_expanded[lr_results_expanded['intention']==intention]
    #         lr_subdf = lr_subdf[lr_subdf['facet_names_values']==f]
    #         lr_subdfgroupby = lr_subdf.groupby('feature')
    #         lr_subdfgroupby = lr_subdfgroupby.mean()
    #         lr_subdfgroupby = lr_subdfgroupby.reset_index()
    #         lr_subdfgroupby['absweight'] = lr_subdfgroupby['weight'].abs()
    #         lr_subdfgroupby = lr_subdfgroupby.sort(['absweight'],ascending=[0])
    #         lr_subdfgroupby['order'] = range(1,len(lr_subdfgroupby.index)+1)
    #         lr_subdfgroupby = lr_subdfgroupby.sort(['feature'],ascending=[1])
    #         ORDERINGS[(f,intention)] = lr_subdfgroupby['order'].tolist()
    #
    #
    # # Stability of weight orderings, for each intention
    # #TODO: Use?
    # for f in list(set(lr_results['facet_names_values'].tolist())):
    #     facet_plot_df = []
    #     for (x,y) in itertools.product(INTENTION_COLUMNS,INTENTION_COLUMNS):
    #         facet_plot_df += [{'intention1':x,'intention2':y,'similarity':stats.kendalltau(ORDERINGS[(f,x)],ORDERINGS[(f,y)])[0]}]
    #     facet_plot_df = pd.DataFrame(facet_plot_df)
    #     facet_plot_df = facet_plot_df.pivot("intention1","intention2","similarity")
    #     plt.figure()
    #     sns_plot = sns.heatmap(facet_plot_df, annot=True,linewidths=.5)
    #     for item in sns_plot.get_yticklabels():
    #         item.set_rotation(0)
    #     for item in sns_plot.get_xticklabels():
    #         item.set_rotation(90)
    #     plt.title('Feature Importance Similarity By Intention - %s'%f)
    #     plt.tight_layout()
    #     plt.savefig(OUTFOLDER+"predictions/images/featureimportancesimilarity_taskfacet_%s.png"%f)
    #
    # #TODO: use?
    # for intention in INTENTION_COLUMNS:
    #     facet_plot_df = []
    #     for (x,y) in itertools.product(list(set(lr_results['facet_names_values'].tolist())),list(set(lr_results['facet_names_values'].tolist()))):
    #         facet_plot_df += [{'facetvalue1':x,'facetvalue2':y,'similarity':stats.kendalltau(ORDERINGS[(x,intention)],ORDERINGS[(y,intention)])[0]}]
    #     facet_plot_df = pd.DataFrame(facet_plot_df)
    #     facet_plot_df = facet_plot_df.pivot("facetvalue1","facetvalue2","similarity")
    #     plt.figure()
    #     sns_plot = sns.heatmap(facet_plot_df, annot=True,linewidths=.5)
    #     for item in sns_plot.get_yticklabels():
    #         item.set_rotation(0)
    #     for item in sns_plot.get_xticklabels():
    #         item.set_rotation(90)
    #     plt.title('Feature Importance Similarity (Kendalltau) For Facet - %s'%f)
    #     plt.tight_layout()
    #     plt.savefig(OUTFOLDER+"predictions/images/featureimportancesimilarity_intention_%s.png"%intention)
    #
    #
    #
    #
    # print lr_results_expanded
    #
    # # Weights of features for each intention
    # for f in list(set(lr_results['facet_names_values'].tolist())):
    #     lr_byfacet = lr_results_expanded[lr_results_expanded['facet_names_values'] == f]
    #     print "FACET",f
    #     for i in INTENTION_COLUMNS:
    #         lr_byfacetintention = lr_results_expanded[lr_results_expanded['intention']==i]
    #         plt.figure()
    #         sns_plot = sns.boxplot(x="feature",y="weight",data=lr_byfacetintention)
    #         for item in sns_plot.get_xticklabels():
    #             item.set_rotation(90)
    #         plt.title('Feature Weights for Intention (%s) By Facet - %s'%(i,f))
    #         plt.tight_layout()
    #         plt.savefig(OUTFOLDER+"predictions/images/featureweight_intention_%s_taskfacet_%s.png"%(i,f))



    # BEGIN TODO 2
    # nrelationships = 0
    # ndifferences = 0
    #
    #
    # # for f in FEATURES:
    # #     for p in PREDICTIONS:
    # #         sns.boxplot(x=p,y=f,data=dataset_tail).get_figure().savefig('/Users/Matt/Desktop/EDA/boxplots/aggregate_dataset_total/dataset_total_%s_by%s.png'%(f,p))
    # #         plt.clf()
    #
    #
    # def percent_overlap(c1, c2):
    #     print("\t Numerator:%d Denominator:%d" % (
    #     sum(((Counter(c1) - Counter(c2)) + (Counter(c2) - Counter(c1))).values()),
    #     sum((Counter(c1) + Counter(c2)).values())))
    #     return sum(((Counter(c1) - Counter(c2)) + (Counter(c2) - Counter(c1))).values()) / sum(
    #         (Counter(c1) + Counter(c2)).values())
    #
    #
    # # for f in FEATURES:
    # #     print("feature",f)
    # #     if f in ['interaction_num','queries_num_session']:
    # #         continue
    # #
    # #     for p in PREDICTIONS:
    # #         predictionvalues =  set(dataset_tail[p].tolist())
    # #         for (pv1,pv2) in itertools.combinations(predictionvalues,2):
    # #             print("\t",p,(pv1,pv2))
    # #             vals1 = dataset_tail[dataset_tail[p]==pv1][f].tolist()
    # #             vals2 = dataset_tail[dataset_tail[p]==pv2][f].tolist()
    # #
    # #             po = percent_overlap(vals1,vals2)
    # #             if len(set(vals1)&set(vals2)) >=2 and po>=0.5:
    # #                 pass
    # #                 # print("\t",p,(pv1,pv2),"overlap! (L1=%d,L2=%d)"%(len(vals1),len(vals2)),len(set(vals1)&set(vals2)),po)
    # #             else:
    # #                 print("\t",p,(pv1,pv2),"overlap! (L1=%d,L2=%d)"%(len(vals1),len(vals2)),len(set(vals1)&set(vals2)),po)
    # #
    # # exit()
    # #
    # #
    # # for f in FEATURES:
    # #     if f in ['interaction_num','queries_num_session']:
    # #         continue
    # #     for p in PREDICTIONS:
    # #
    # #         predictionvalues =  set(dataset_tail[p].tolist())
    # #         predictionstring = ''
    # #         for (pv1,pv2) in itertools.combinations(predictionvalues,2):
    # #             vals1 = dataset_tail[dataset_tail[p]==pv1][f].tolist()
    # #             vals2 = dataset_tail[dataset_tail[p]==pv2][f].tolist()
    # #             (stat,pval) = stats.mannwhitneyu(vals1,vals2)
    # #             diff = ''
    # #             if pval < .05:
    # #                 diff = '*'
    # #                 predictionstring += '(%s,%s,%s)'%(pv1,pv2,diff)
    # #             elif pval < .01:
    # #                 diff = '**'
    # #                 predictionstring += '(%s,%s,%s)'%(pv1,pv2,diff)
    # #             elif pval < .001:
    # #                 diff = '***'
    # #                 predictionstring += '(%s,%s,%s)'%(pv1,pv2,diff)
    # #
    # #
    # #         sns.boxplot(x='interaction_num',y=f,hue=p,data=dataset_tail).get_figure().savefig('/Users/Matt/Desktop/EDA/boxplots/byquerynum_dataset_tail/dataset_tail_%s_by%s.png'%(f,p))
    # #         plt.clf()
    #
    #
    # for f in FEATURES:
    #     for p in PREDICTIONS:
    #         sns.boxplot(x=p, y=f, data=dataset_tail)
    #         # .get_figure()
    #         # .savefig('/Users/Matt/Desktop/EDA/boxplots/byquerynum_dataset_total/dataset_total_%s_by%s.png'%(f,p))
    #         sns.swarmplot(x=p, y=f, data=dataset_tail).get_figure().savefig(
    #             '/Users/Matt/Desktop/EDA/boxplots/aggregate_dataset_tail/dataset_tail_%s_by%s.png' % (f, p))
    #         plt.clf()
    #
    # for f in FEATURES:
    #     if f in ['interaction_num', 'queries_num_session']:
    #         continue
    #     for p in PREDICTIONS:
    #         sns.boxplot(x='interaction_num', y=f, hue=p, data=dataset_tail)
    #         # .get_figure()
    #         # .savefig('/Users/Matt/Desktop/EDA/boxplots/byquerynum_dataset_total/dataset_total_%s_by%s.png'%(f,p))
    #         sns.swarmplot(x='interaction_num', y=f, hue=p, data=dataset_tail).get_figure().savefig(
    #             '/Users/Matt/Desktop/EDA/boxplots/byquerynum_dataset_tail/dataset_tail_%s_by%s.png' % (f, p))
    #         plt.clf()
    #
    # dataset_tail = dataset
    # for f in FEATURES:
    #     for p in PREDICTIONS:
    #         sns.boxplot(x=p, y=f, data=dataset_tail)
    #         # .get_figure()
    #         # .savefig('/Users/Matt/Desktop/EDA/boxplots/byquerynum_dataset_total/dataset_total_%s_by%s.png'%(f,p))
    #         sns.swarmplot(x=p, y=f, data=dataset_tail).get_figure().savefig(
    #             '/Users/Matt/Desktop/EDA/boxplots/aggregate_dataset_total/dataset_total_%s_by%s.png' % (f, p))
    #         plt.clf()
    #
    # for f in FEATURES:
    #     if f in ['interaction_num', 'queries_num_session']:
    #         continue
    #     for p in PREDICTIONS:
    #         sns.boxplot(x='interaction_num', y=f, hue=p, data=dataset_tail)
    #         # .get_figure()
    #         # .savefig('/Users/Matt/Desktop/EDA/boxplots/byquerynum_dataset_total/dataset_total_%s_by%s.png'%(f,p))
    #         sns.swarmplot(x='interaction_num', y=f, hue=p, data=dataset_tail).get_figure().savefig(
    #             '/Users/Matt/Desktop/EDA/boxplots/byquerynum_dataset_total/dataset_total_%s_by%s.png' % (f, p))
    #         plt.clf()
    # exit()
    #
    # dataset_tail = dataset
    # for f in FEATURES:
    #     if f in ['interaction_num', 'queries_num_session']:
    #         continue
    #     for p in PREDICTIONS:
    #         sns.boxplot(x='interaction_num', y=f, hue=p, data=dataset_tail).get_figure().savefig(
    #             '/Users/Matt/Desktop/EDA/boxplots/byquerynum_dataset_total/dataset_total_%s_by%s.png' % (f, p))
    #         plt.clf()
    # exit()


    # for p in PREDICTIONS:
    #
    #     predictionvalues = set(prediction_dataset[p].tolist())
    #     for (pv1, pv2) in itertools.combinations(predictionvalues, 2):
    #
    #         for f in FEATURES:
    #             nrelationships += 1
    #             vals1 = prediction_dataset[prediction_dataset[p] == pv1][f].tolist()
    #             vals2 = prediction_dataset[prediction_dataset[p] == pv2][f].tolist()
    #             try:
    #                 (stat, pval) = stats.mannwhitneyu(vals1, vals2)
    #                 diff = ''
    #                 if pval < .05:
    #                     diff = '*'
    #                     ndifferences += 1
    #                     print("%s differs for %s,%s" % (f, pv1, pv2))
    #                 elif pval < .01:
    #                     diff = '**'
    #                 elif pval < .001:
    #                     diff = '***'
    #             except ValueError:
    #                 pass
    #                 # print("Values the same for (%s,%s,%s,%s)"%(p,pv1,pv2,f))
    #
    # print("percent differences %f" % (ndifferences / nrelationships), nrelationships, ndifferences)
    # exit()
