from __future__ import print_function
import random
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
N_DOWNSAMPLED_SAMPLES = 20
N_RANDOM_SAMPLES = 20
N_RANDOM_BASE = 100
N_RANDOM_INCREMENTS = 100

NSF_N_TOTAL = 697
SAL_N_TOTAL = 594
EOPN_N_TOTAL = 1274


nsf_exclusive = nsf_data_exclusive = pd.read_csv('/Users/Matt/Desktop/nsf_exclusive_scaled.csv')
sal_exclusive = sal_data_exclusive = pd.read_csv('/Users/Matt/Desktop/sal_exclusive_scaled.csv')
eopn_exclusive = eopn_data_exclusive = pd.read_csv('/Users/Matt/Desktop/eopn_exclusive_scaled.csv')


first_index = NSF_N_TOTAL
second_index = NSF_N_TOTAL+SAL_N_TOTAL
nsf_sal_data_allcolumns = pd.read_csv('/Users/Matt/Desktop/nsf_sal_allcolumns_scaled.csv')
nsf_1 = nsf_sal_data_allcolumns[0:first_index]
sal_1 = nsf_sal_data_allcolumns[first_index:]
print(nsf_1.head(1).isnull())
print(nsf_1.tail(1).isnull())
print(sal_1.head(1).isnull())
print(sal_1.tail(1).isnull())




first_index = NSF_N_TOTAL
second_index = NSF_N_TOTAL+EOPN_N_TOTAL
nsf_eopn_data_allcolumns = pd.read_csv('/Users/Matt/Desktop/nsf_eopn_allcolumns_scaled.csv')
nsf_2 = nsf_eopn_data_allcolumns[0:first_index]
eopn_2 = nsf_eopn_data_allcolumns[first_index:]
print(len(nsf_2.index),len(eopn_2.index))
print(nsf_2.head(1))
print(nsf_2.tail(1))
print(eopn_2.head(1))
print(eopn_2.tail(1))


first_index = SAL_N_TOTAL
second_index = SAL_N_TOTAL+EOPN_N_TOTAL
sal_eopn_data_allcolumns = pd.read_csv('/Users/Matt/Desktop/sal_eopn_allcolumns_scaled.csv')
sal_3 = sal_eopn_data_allcolumns[0:first_index]
eopn_3 = sal_eopn_data_allcolumns[first_index:]
print(len(sal_3.index),len(eopn_3.index))
print(sal_3.head(1))
print(sal_3.tail(1))
print(eopn_3.head(1))
print(eopn_3.tail(1))


first_index = NSF_N_TOTAL
second_index = NSF_N_TOTAL+SAL_N_TOTAL
third_index = NSF_N_TOTAL+SAL_N_TOTAL+EOPN_N_TOTAL
nsf_sal_eopn_data_allcolumns = pd.read_csv('/Users/Matt/Desktop/nsf_sal_eopn_allcolumns_scaled.csv')
nsf_4 = nsf_sal_eopn_data_allcolumns[0:first_index]
sal_4 = nsf_sal_eopn_data_allcolumns[first_index:second_index]
eopn_4 = nsf_sal_eopn_data_allcolumns[second_index:]
print(len(nsf_4.index),len(sal_4.index),len(eopn_4.index))
print(nsf_4.head(1))
print(nsf_4.tail(1))
print(sal_4.head(1))
print(sal_4.tail(1))
print(eopn_4.head(1))
print(eopn_4.tail(1))



SAMP_PERCENTAGES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
n = 0


print("randommissing")

for (dataname,data_dup) in [
        ('nsf',nsf_exclusive),
        ('sal',sal_exclusive),
        ('eopn',eopn_exclusive),
    ]:
    print(dataname)
    for downsample_num in range(1,N_DOWNSAMPLED_SAMPLES+1):
        for perc in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            data = data_dup.copy(deep=True)
            df_train = data.sample(frac=0.8)
            smp_test = data[~data.isin(df_train)].dropna(how = 'all')

            ix = [(row, col) for row in range(df_train.shape[0]) for col in range(df_train.shape[1])]
            for row, col in random.sample(ix, int(round(perc*len(ix)))):
                df_train.iloc[row, col] = None


            # ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
            # for row, col in random.sample(ix, int(round(perc*len(ix)))):
            #     data.iloc[row, col] = None
            #
            # # print("\t",sum(data.isna().sum()),sum(data.notna().sum()))
            # df_train = data.sample(frac=0.8)
            # smp_test = data[~data.isin(df_train)].dropna(how = 'all')

            # print("\t",sum(data.isna().sum()),sum(data.notna().sum()))

            n+=1
            print("\t",len(df_train.columns.values))
            if dataname=='nsf':
                assert(len(df_train.columns.values)==len(smp_test.columns.values)==15)
            elif dataname=='sal':
                assert(len(df_train.columns.values)==len(smp_test.columns.values)==12)
            else:
                assert(len(df_train.columns.values)==len(smp_test.columns.values)==11)

            df_train.to_csv('/Users/Matt/Desktop/Experiment/data/interim/randommissing/train/sample%d_train_%s_perc_%d_test_%s_perc_%d_percmissing_%d.csv'%(downsample_num,dataname,int(0.8*100),dataname,int(0.2*100),int(perc*100)))
            smp_test.to_csv('/Users/Matt/Desktop/Experiment/data/interim/randommissing/test/sample%d_train_%s_perc_%d_test_%s_perc_%d_percmissing_%d.csv'%(downsample_num,dataname,int(0.8*100),dataname,int(0.2*100),int(perc*100)))
        # print("")
        #
        #

print("exclusive")
for (dataname,data) in [
        ('nsf',nsf_exclusive),
        ('sal',sal_exclusive),
        ('eopn',eopn_exclusive),
    ]:
    print(dataname)
    for downsample_num in range(1,N_DOWNSAMPLED_SAMPLES+1):
        for perc in SAMP_PERCENTAGES:
            df_train = data.sample(frac=perc)
            smp_test = data[~data.isin(df_train)].dropna(how = 'all')
            n+=1

            if dataname=='nsf':
                assert(len(df_train.columns.values)==len(smp_test.columns.values)==15)
            elif dataname=='sal':
                assert(len(df_train.columns.values)==len(smp_test.columns.values)==12)
            else:
                assert(len(df_train.columns.values)==len(smp_test.columns.values)==11)
            print("\t",len(df_train.index),len(smp_test.index))
            p = int(perc*100)
            df_train.to_csv('/Users/Matt/Desktop/Experiment/data/interim/exclusive/train/sample%d_train_%s_perc_%d_test_%s_perc_%d.csv'%(downsample_num,dataname,p,dataname,100-p))
            smp_test.to_csv('/Users/Matt/Desktop/Experiment/data/interim/exclusive/test/sample%d_train_%s_perc_%d_test_%s_perc_%d.csv'%(downsample_num,dataname,p,dataname,100-p))






print("allcolumns")
for (training_sets,training_set_names,test_set_name,test_set) in [
        ((sal_1,),('sal',),'nsf',nsf_1),
        ((nsf_1,),('nsf',),'sal',sal_1),

        ((eopn_2,),('eopn',),'nsf',nsf_2),
        ((nsf_2,),('nsf',),'eopn',eopn_2),

        ((sal_3,),('sal',),'eopn',eopn_3),
        ((eopn_3,),('eopn',),'sal',sal_3),

        ((sal_4,eopn_4),('sal','eopn'),'nsf',nsf_4),
        ((nsf_4,eopn_4),('nsf','eopn'),'sal',sal_4),
        ((nsf_4,sal_4),('nsf','sal'),'eopn',eopn_4),
    ]:
    print(training_set_names,test_set_name)
    for downsample_num in range(1,N_DOWNSAMPLED_SAMPLES+1):
        for perc in SAMP_PERCENTAGES:
            print("\t",perc)
            df_train = []
            for t in training_sets:
                smp = t.sample(frac=perc)
                # if downsample_num==1 and perc in [0.5,1.0]:
                #     print(len(smp.index))
                df_train += [smp]
                if downsample_num==1:
                    print("\t","train",len(smp.index),len(t.index))
            smp = test_set.sample(frac=0.8)
            if downsample_num==1:
                print("\t","test",len(smp.index),len(test_set.index))
            smp_test = test_set[~test_set.isin(smp)].dropna(how = 'all')

            df_train += [smp]
            df_train = pd.concat(df_train)
            train_prefix = '_'.join(training_set_names)
            n+=1
            if downsample_num==1:
                print("\t","test sum",len(df_train.index),len(smp_test.index),len(test_set.index))
            assert(len(df_train.columns.values)==len(smp_test.columns.values)==15)
            df_train.to_csv('/Users/Matt/Desktop/Experiment/data/interim/allcolumns/train/sample%d_train_%s_perc_%d_test_%s_perc_%d.csv'%(downsample_num,train_prefix,int(perc*100),test_set_name,80))
            smp_test.to_csv('/Users/Matt/Desktop/Experiment/data/interim/allcolumns/test/sample%d_train_%s_perc_%d_test_%s_perc_%d.csv'%(downsample_num,train_prefix,int(perc*100),test_set_name,80))

        for perc in SAMP_PERCENTAGES:
            df_train = []
            print("\t",perc)
            for t in training_sets:
                smp = t.sample(frac=1.0)
                # if downsample_num==1 and perc in [0.5,1.0]:
                #     print(len(smp.index))
                df_train += [smp]
                if downsample_num==1:
                    print("\t","train",len(smp.index),len(t.index))
            smp = test_set.sample(frac=perc)
            if downsample_num==1:
                print("\t","test",len(smp.index),len(test_set.index))
            smp_test = test_set[~test_set.isin(smp)].dropna(how = 'all')

            df_train += [smp]
            df_train = pd.concat(df_train)
            train_prefix = '_'.join(training_set_names)
            n+=1
            if downsample_num==1:
                print("\t","test sum",len(df_train.index),len(smp_test.index),len(test_set.index))
            assert(len(df_train.columns.values)==len(smp_test.columns.values)==15)
            df_train.to_csv('/Users/Matt/Desktop/Experiment/data/interim/allcolumns/train/sample%d_train_%s_perc_%d_test_%s_perc_%d.csv'%(downsample_num,train_prefix,100,test_set_name,int(perc*100)))
            smp_test.to_csv('/Users/Matt/Desktop/Experiment/data/interim/allcolumns/test/sample%d_train_%s_perc_%d_test_%s_perc_%d.csv'%(downsample_num,train_prefix,100,test_set_name,int(perc*100)))











print(n)
