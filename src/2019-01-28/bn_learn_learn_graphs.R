library(bnlearn)
library(Rgraphviz)


N_DOWNSAMPLED_SAMPLES <- 20
N_RANDOM_SAMPLES <- 20
#N_DOWNSAMPLED_SAMPLES <- 50
#N_RANDOM_SAMPLES <- 50
N_RANDOM_BASE <- 100
N_RANDOM_INCREMENTS <- 100
N_BOOTSTRAP_ITERS <- 20
DATA_DIRECTORY <- '/Users/Matt/Desktop/Experiment/data/interim/'
GRAPH_DIRECTORY <- '/Users/Matt/Desktop/Experiment/data/interim/trained_structure_graphs/'
EVAL_METRICS_DIRECTORY <- '/Users/Matt/Desktop/Experiment/data/interim/structure_graph_metrics/'

RANDOM_SAMPS <- c(10,20,30,40,50,60,70,80,90)
PERC_MISSING < c(0,10,20,30,40,50,60,70,80)
#TODO: Full parameters
train_datasets_allcolumns <- c(
  'nsf',
  'sal',
  
  'eopn',
  'nsf',
  'sal',
  'eopn',
  'sal_eopn',
  'nsf_eopn',
  'nsf_sal'
)

test_datasets_allcolumns <- c(
  'sal',
  'nsf',
  
  'nsf',
  'eopn',
  'eopn',
  'sal',
  'nsf',
  'sal',
  'eopn'
)


train_datasets_exclusive <- c(
  'nsf',
  'sal',
  'eopn'
)


load_dataset <- function(subdirectory,train_or_test,sample_number,training_name,percent_training,test_name,percent_test){
  
  subdirectory_string<-paste(subdirectory,'/',train_or_test,'/',sep='')
  samplestring<-paste('sample',sample_number,'_',sep='')
  trainingdatastring<-paste('train_',training_name,'_',sep='')
  trainpercentstring<-paste('perc_',percent_training,'_',sep='')
  testdatastring<-paste('test_',test_name,'_',sep='')
  testpercentstring<-paste('perc_',percent_test,sep='')
  print(paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,'.csv',sep=''))
  data <- read.csv(paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,'.csv',sep=''),row.names=1)
  data<-data[ , -which(names(data) %in% c('Unnamed..0'))]
  #data <- read.csv(paste(DATA_DIRECTORY,subdirectory_string,,sep=''),row.names=1)

  return(data)
}


load_dataset_percmissing <- function(subdirectory,train_or_test,sample_number,training_name,percent_training,test_name,percent_test,percent_missing){
  
  subdirectory_string<-paste(subdirectory,'/',train_or_test,'/',sep='')
  samplestring<-paste('sample',sample_number,'_',sep='')
  trainingdatastring<-paste('train_',training_name,'_',sep='')
  trainpercentstring<-paste('perc_',percent_training,'_',sep='')
  testdatastring<-paste('test_',test_name,'_',sep='')
  testpercentstring<-paste('perc_',percent_test,'_',sep='')
  percentmissingstring<-paste('percmissing_',percent_missing,sep='')
  print(paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,percentmissingstring,'.csv',sep=''))
  data <- read.csv(paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,percentmissingstring,'.csv',sep=''),row.names=1)
  data<-data[ , -which(names(data) %in% c('Unnamed..0'))]
  #data <- read.csv(paste(DATA_DIRECTORY,subdirectory_string,,sep=''),row.names=1)
  
  return(data)
}

save_graph <- function(graph,subdirectory,train_or_test,sample_number,training_name,percent_training,test_name,percent_test){
  
  subdirectory_string<-paste('trained_structure_graphs','/',subdirectory,'/',sep='')
  samplestring<-paste('sample',sample_number,'_',sep='')
  trainingdatastring<-paste('train_',training_name,'_',sep='')
  trainpercentstring<-paste('perc_',percent_training,'_',sep='')
  testdatastring<-paste('test_',test_name,'_',sep='')
  testpercentstring<-paste('perc_',percent_test,sep='')
  print(paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,'.rds',sep=''))
  filename<-paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,'.rds',sep='')
  saveRDS(graph,filename)
  #data <- read.csv(,row.names=1)
  #data<-data[ , -which(names(data) %in% c('Unnamed..0'))]
  #data <- read.csv(paste(DATA_DIRECTORY,subdirectory_string,,sep=''),row.names=1)
  
  #return(data)
}


save_graph_percmissing <- function(graph,subdirectory,train_or_test,sample_number,training_name,percent_training,test_name,percent_test,percent_missing){
  
  subdirectory_string<-paste('trained_structure_graphs','/',subdirectory,'/',sep='')
  samplestring<-paste('sample',sample_number,'_',sep='')
  trainingdatastring<-paste('train_',training_name,'_',sep='')
  trainpercentstring<-paste('perc_',percent_training,'_',sep='')
  testdatastring<-paste('test_',test_name,'_',sep='')
  testpercentstring<-paste('perc_',percent_test,'_',sep='')
  percentmissingstring<-paste('percmissing_',percent_missing,sep='')
  print(paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,percentmissingstring,'.rds',sep=''))
  #data <- read.csv(paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,percentmissingstring,'.csv',sep=''),row.names=1)
  filename<-paste(DATA_DIRECTORY,subdirectory_string,samplestring,trainingdatastring,trainpercentstring,testdatastring,testpercentstring,percentmissingstring,'.rds',sep='')
  saveRDS(graph,filename)
  #data<-data[ , -which(names(data) %in% c('Unnamed..0'))]
  #data <- read.csv(paste(DATA_DIRECTORY,subdirectory_string,,sep=''),row.names=1)
  
  #return(data)
}

retransform_data <- function(data){
  data[] <- lapply(data, function(x) {
    if(is.factor(x) || is.integer(x)) as.numeric(as.character(x)) else x
  })
  
  #no scaling - already done
  return(data)
}


create_row <- function(training_set_name,
                       test_set_name,
                       sample_number,
                       percent_training,
                       percent_test,
                       algorithm,
                        feature,
                       AIC_score,BIC_score,logLik_score,
                       percent_missing
                       ){
  row <- data.frame(training_set=training_set_name,
                    test_set=test_set_name,
                    sample_number=sample_number,
                    percent_training=percent_training,
                    percent_test=percent_test,
                    algorithm=algorithm,
                    feature=feature,
                    AIC=AIC_score,
                    BIC=BIC_score,
                    logLik=logLik_score,
                    percent_missing=percent_missing
  )
  return(row)
}

omit_missing_values <- function(train_data,test_data){
  train_data_copy <- data.frame(train_data)
  test_data_copy <- data.frame(test_data)
  for(cn in colnames(train_data_copy)){
    num_nas<-sum(is.na(train_data_copy[,cn]))
    num_nas_test<-sum(is.na(test_data_copy[,cn]))
    if(num_nas==nrow(train_data_copy)){
      train_data_copy<- train_data_copy[ , !(names(train_data_copy) %in% c(cn))]
      test_data_copy<- test_data_copy[ , !(names(test_data_copy) %in% c(cn))]
    }else if(num_nas_test==nrow(test_data_copy)){
      mn<- mean(train_data_copy[,cn],na.rm=TRUE)
      rndrow<-sample(nrow(test_data_copy), 1)
      test_data_copy[rndrow , cn]<- mn
    }  
  }
  
  return(list(train_data_copy,test_data_copy));
}

###
#
# Experiment 1
#
###


### Experiment 1.1

i<-0
dataset_metrics_frame = data.frame();
for(col in train_datasets_allcolumns){
  i<-i+1
  train_dataset_name<-train_datasets_allcolumns[i]
  test_dataset_name<-test_datasets_allcolumns[i]
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  #n_downsamps<- 1
  PERC_MISSING <- c(0,10,20,30)
  
  for(sample_percent in c(100)){
    sample_complement<-80
    for(downsamp_n in (1:n_downsamps)){
      subdir<- 'allcolumns'
      train_data<-load_dataset(subdir,'train',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      train_data <-retransform_data(train_data)
      test_data<-load_dataset(subdir,'test',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      test_data <-retransform_data(test_data)
      print("BEFORE FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      
      data_omitcols<-omit_missing_values(train_data,test_data)
      train_data <- data_omitcols[[1]] 
      test_data <- data_omitcols[[2]] 
      print("AFTER FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      #print(colnames(train_data))
      
      print("Train")
      s<-structural.em(train_data,return.all=TRUE)
      save_graph(s,subdir,'train',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      bnobj_fitted<-s$fitted
      bnobj<-bn.net(bnobj_fitted)
      #test_data_imputed<-test_data
      test_data_imputed <- impute(bnobj_fitted,test_data)
      print("Done training")
      
      print("LL")
      ll_bynode<-logLik(bnobj,test_data_imputed,by.node=TRUE)
      ll_all<-sum(ll_bynode[is.finite(ll_bynode)])
      
      print("AIC")
      aic_bynode<-AIC(bnobj,test_data_imputed,by.node=TRUE)
      aic_all<-sum(aic_bynode[is.finite(aic_bynode)])
      
      print("BIC")
      bic_bynode<-BIC(bnobj,test_data_imputed,by.node=TRUE)
      bic_all<-sum(bic_bynode[is.finite(bic_bynode)])
      
      dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                   create_row(
                                     train_dataset_name,
                                     test_dataset_name,
                                     downsamp_n,
                                     sample_percent,
                                     sample_complement,
                                     'em',
                                     'all',
                                     aic_all,bic_all,ll_all,
                                     0
                                   ));
      for(n in names(ll_bynode)){
        ll_node<-ll_bynode[n]
        aic_node<-aic_bynode[n]
        bic_node<-bic_bynode[n]
        dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                     create_row(
                                       train_dataset_name,
                                       test_dataset_name,
                                       downsamp_n,
                                       sample_percent,
                                       sample_complement,
                                       'em',
                                       n,
                                       aic_node,
                                       bic_node,
                                       ll_node,
                                       0
                                     ));
        
      }
      print("Done evaluation")
    }
  }
  
  
  #print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}

write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'exp1_combineddataset_metrics.csv',sep=''))


### Experiment 1.2 - single dataset only
i<-0
dataset_metrics_frame = data.frame();
for(col in train_datasets_exclusive){
  i<-i+1
  train_dataset_name<-train_datasets_exclusive[i]
  test_dataset_name<-train_datasets_exclusive[i]
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  PERC_MISSING <- c(0,10,20,30)
  
  for(sample_percent in RANDOM_SAMPS){
    sample_complement<-100-sample_percent
    for(downsamp_n in (1:n_downsamps)){
      subdir<- 'exclusive'
      train_data<-load_dataset(subdir,'train',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      train_data <-retransform_data(train_data)
      test_data<-load_dataset(subdir,'test',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      test_data <-retransform_data(test_data)
      print("BEFORE FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      
      data_omitcols<-omit_missing_values(train_data,test_data)
      train_data <- data_omitcols[[1]] 
      test_data <- data_omitcols[[2]] 
      print("AFTER FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      #print(colnames(train_data))
      
      print("Train")
      s<-mmhc(train_data)
      save_graph(s,subdir,'train',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      #s<-structural.em(train_data,return.all=TRUE)
      bnobj_fitted<-s$fitted
      bnobj<-s
      test_data_imputed<-test_data
      #test_data_imputed <- impute(bnobj_fitted,test_data)
      print("Done training")
      
      print("LL")
      ll_bynode<-logLik(bnobj,test_data_imputed,by.node=TRUE)
      ll_all<-sum(ll_bynode[is.finite(ll_bynode)])
      
      print("AIC")
      aic_bynode<-AIC(bnobj,test_data_imputed,by.node=TRUE)
      aic_all<-sum(aic_bynode[is.finite(aic_bynode)])
      
      print("BIC")
      bic_bynode<-BIC(bnobj,test_data_imputed,by.node=TRUE)
      bic_all<-sum(bic_bynode[is.finite(bic_bynode)])
      
      dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                   create_row(
                                     train_dataset_name,
                                     test_dataset_name,
                                     downsamp_n,
                                     sample_percent,
                                     sample_complement,
                                     'mmhc',
                                     'all',
                                     aic_all,bic_all,ll_all,
                                     0
                                   ));
      for(n in names(ll_bynode)){
        ll_node<-ll_bynode[n]
        aic_node<-aic_bynode[n]
        bic_node<-bic_bynode[n]
        dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                     create_row(
                                       train_dataset_name,
                                       test_dataset_name,
                                       downsamp_n,
                                       sample_percent,
                                       sample_complement,
                                       'mmhc',
                                       n,
                                       aic_node,
                                       bic_node,
                                       ll_node,
                                       0
                                     ));
        
      }
      print("Done evaluation")
    }
  }
  
  
  #print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}

write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'exp1_onedataset_metrics.csv',sep=''))
write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'exp2_onedataset_metrics.csv',sep=''))

###
#
# Experiment 2
#
###

i<-0
dataset_metrics_frame = data.frame();
for(col in train_datasets_allcolumns){
  i<-i+1
  train_dataset_name<-train_datasets_allcolumns[i]
  test_dataset_name<-test_datasets_allcolumns[i]
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  #n_downsamps<- 1
  PERC_MISSING <- c(0,10,20,30)
  
  for(sample_percent in c(10,20,30,40,50,60,70,80,90,100)){
    sample_complement<-80
    for(downsamp_n in (1:n_downsamps)){
      subdir<- 'allcolumns'
      train_data<-load_dataset(subdir,'train',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      train_data <-retransform_data(train_data)
      test_data<-load_dataset(subdir,'test',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      test_data <-retransform_data(test_data)
      print("BEFORE FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      
      data_omitcols<-omit_missing_values(train_data,test_data)
      train_data <- data_omitcols[[1]] 
      test_data <- data_omitcols[[2]] 
      print("AFTER FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      #print(colnames(train_data))
      
      print("Train")
      s<-structural.em(train_data,return.all=TRUE)
      save_graph(s,subdir,'train',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      #s<-mmhc(train_data)
      bnobj_fitted<-s$fitted
      bnobj<-bn.net(bnobj_fitted)
      #bnobj<-s
      #test_data_imputed<-test_data
      test_data_imputed <- impute(bnobj_fitted,test_data)
      print("Done training")
      
      print("LL")
      ll_bynode<-logLik(bnobj,test_data_imputed,by.node=TRUE)
      ll_all<-sum(ll_bynode[is.finite(ll_bynode)])
      
      print("AIC")
      aic_bynode<-AIC(bnobj,test_data_imputed,by.node=TRUE)
      aic_all<-sum(aic_bynode[is.finite(aic_bynode)])
      
      print("BIC")
      bic_bynode<-BIC(bnobj,test_data_imputed,by.node=TRUE)
      bic_all<-sum(bic_bynode[is.finite(bic_bynode)])
      
      dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                   create_row(
                                     train_dataset_name,
                                     test_dataset_name,
                                     downsamp_n,
                                     sample_percent,
                                     sample_complement,
                                     'em',
                                     'all',
                                     aic_all,bic_all,ll_all,
                                     0
                                   ));
      for(n in names(ll_bynode)){
        ll_node<-ll_bynode[n]
        aic_node<-aic_bynode[n]
        bic_node<-bic_bynode[n]
        dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                     create_row(
                                       train_dataset_name,
                                       test_dataset_name,
                                       downsamp_n,
                                       sample_percent,
                                       sample_complement,
                                       'em',
                                       n,
                                       aic_node,
                                       bic_node,
                                       ll_node,
                                       0
                                     ));
        
      }
      print("Done evaluation")
    }
  }
  
  
  #print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}

write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'exp2_combineddataset_metrics.csv',sep=''))




#EXP 2.2

i<-0
dataset_metrics_frame = data.frame();
for(col in train_datasets_allcolumns){
  i<-i+1
  train_dataset_name<-train_datasets_allcolumns[i]
  test_dataset_name<-test_datasets_allcolumns[i]
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  #n_downsamps<- 1
  PERC_MISSING <- c(0,10,20,30)
  
  for(sample_percent in c(10,20,30,40,50,60,70,80,90)){
    sample_complement<-100
    for(downsamp_n in (1:n_downsamps)){
      subdir<- 'allcolumns'
      train_data<-load_dataset(subdir,'train',downsamp_n,train_dataset_name,sample_complement,test_dataset_name,sample_percent);
      train_data <-retransform_data(train_data)
      test_data<-load_dataset(subdir,'test',downsamp_n,train_dataset_name,sample_complement,test_dataset_name,sample_percent);
      test_data <-retransform_data(test_data)
      print("BEFORE FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      
      data_omitcols<-omit_missing_values(train_data,test_data)
      train_data <- data_omitcols[[1]] 
      test_data <- data_omitcols[[2]] 
      print("AFTER FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      #print(colnames(train_data))
      
      print("Train")
      s<-structural.em(train_data,return.all=TRUE)
      save_graph(s,subdir,'train',downsamp_n,train_dataset_name,sample_complement,test_dataset_name,sample_percent);
      #s<-mmhc(train_data)
      bnobj_fitted<-s$fitted
      bnobj<-bn.net(bnobj_fitted)
      #bnobj<-s
      #test_data_imputed<-test_data
      test_data_imputed <- impute(bnobj_fitted,test_data)
      print("Done training")
      
      print("LL")
      ll_bynode<-logLik(bnobj,test_data_imputed,by.node=TRUE)
      ll_all<-sum(ll_bynode[is.finite(ll_bynode)])
      
      print("AIC")
      aic_bynode<-AIC(bnobj,test_data_imputed,by.node=TRUE)
      aic_all<-sum(aic_bynode[is.finite(aic_bynode)])
      
      print("BIC")
      bic_bynode<-BIC(bnobj,test_data_imputed,by.node=TRUE)
      bic_all<-sum(bic_bynode[is.finite(bic_bynode)])
      
      dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                   create_row(
                                     train_dataset_name,
                                     test_dataset_name,
                                     downsamp_n,
                                     sample_complement,
                                     sample_percent,
                                     
                                     'em',
                                     'all',
                                     aic_all,bic_all,ll_all,
                                     0
                                   ));
      for(n in names(ll_bynode)){
        ll_node<-ll_bynode[n]
        aic_node<-aic_bynode[n]
        bic_node<-bic_bynode[n]
        dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                     create_row(
                                       train_dataset_name,
                                       test_dataset_name,
                                       downsamp_n,
                                       sample_percent,
                                       sample_complement,
                                       'em',
                                       n,
                                       aic_node,
                                       bic_node,
                                       ll_node,
                                       0
                                     ));
        
      }
      print("Done evaluation")
    }
  }
  
  
  #print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}

write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'exp2_combineddataset_metrics_2.csv',sep=''))



### Experiment 2.3 - single dataset only
i<-0
dataset_metrics_frame = data.frame();
for(col in train_datasets_exclusive){
  i<-i+1
  train_dataset_name<-train_datasets_exclusive[i]
  test_dataset_name<-train_datasets_exclusive[i]
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  PERC_MISSING <- c(0,10,20,30)
  
  for(sample_percent in c(10,20,30,40,50,60,70,80,90)){
    sample_complement<-100-sample_percent
    for(downsamp_n in (1:n_downsamps)){
      subdir<- 'exclusive'
      train_data<-load_dataset(subdir,'train',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      train_data <-retransform_data(train_data)
      test_data<-load_dataset(subdir,'test',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      test_data <-retransform_data(test_data)
      print("BEFORE FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      
      data_omitcols<-omit_missing_values(train_data,test_data)
      train_data <- data_omitcols[[1]] 
      test_data <- data_omitcols[[2]] 
      print("AFTER FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      #print(colnames(train_data))
      
      print("Train")
      s<-mmhc(train_data)
      save_graph(s,subdir,'train',downsamp_n,train_dataset_name,sample_percent,test_dataset_name,sample_complement);
      #s<-structural.em(train_data,return.all=TRUE)
      bnobj_fitted<-s$fitted
      bnobj<-s
      test_data_imputed<-test_data
      #test_data_imputed <- impute(bnobj_fitted,test_data)
      print("Done training")
      
      print("LL")
      ll_bynode<-logLik(bnobj,test_data_imputed,by.node=TRUE)
      ll_all<-sum(ll_bynode[is.finite(ll_bynode)])
      
      print("AIC")
      aic_bynode<-AIC(bnobj,test_data_imputed,by.node=TRUE)
      aic_all<-sum(aic_bynode[is.finite(aic_bynode)])
      
      print("BIC")
      bic_bynode<-BIC(bnobj,test_data_imputed,by.node=TRUE)
      bic_all<-sum(bic_bynode[is.finite(bic_bynode)])
      
      dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                   create_row(
                                     train_dataset_name,
                                     test_dataset_name,
                                     downsamp_n,
                                     sample_percent,
                                     sample_complement,
                                     'mmhc',
                                     'all',
                                     aic_all,bic_all,ll_all,
                                     0
                                   ));
      for(n in names(ll_bynode)){
        ll_node<-ll_bynode[n]
        aic_node<-aic_bynode[n]
        bic_node<-bic_bynode[n]
        dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                     create_row(
                                       train_dataset_name,
                                       test_dataset_name,
                                       downsamp_n,
                                       sample_percent,
                                       sample_complement,
                                       'mmhc',
                                       n,
                                       aic_node,
                                       bic_node,
                                       ll_node,
                                       0
                                     ));
        
      }
      print("Done evaluation")
    }
  }
  
  
  #print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}

write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'exp2_onedataset_metrics.csv',sep=''))






###
#
# Experiment 3
#
###

i<-0
dataset_metrics_frame = data.frame();
for(col in train_datasets_exclusive){
  i<-i+1
  train_dataset_name<-train_datasets_exclusive[i]
  test_dataset_name<-train_datasets_exclusive[i]
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  PERC_MISSING <- c(0,10,20,30,40,50,60,70,80)
  
  for(percmissing in PERC_MISSING){
    for(downsamp_n in (1:n_downsamps)){
      subdir<- 'randommissing'
      train_data<-load_dataset_percmissing(subdir,'train',downsamp_n,train_dataset_name,80,test_dataset_name,20,percmissing);
      train_data <-retransform_data(train_data)
      test_data<-load_dataset_percmissing(subdir,'test',downsamp_n,train_dataset_name,80,test_dataset_name,20,percmissing);
      test_data <-retransform_data(test_data)
      print("BEFORE FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      
      data_omitcols<-omit_missing_values(train_data,test_data)
      train_data <- data_omitcols[[1]] 
      test_data <- data_omitcols[[2]] 
      print("AFTER FILTER")
      #print(nrow(train_data))
      print(ncol(train_data))
      #print(nrow(test_data))
      print(ncol(test_data))
      #print(colnames(train_data))
      
      print("Train")
      s<-structural.em(train_data,return.all=TRUE)
      save_graph_percmissing(s,subdir,'train',downsamp_n,train_dataset_name,80,test_dataset_name,20,percmissing);
      bnobj_fitted<-s$fitted
      bnobj<-bn.net(bnobj_fitted)
      test_data_imputed<-test_data
      #test_data_imputed <- impute(bnobj_fitted,test_data)
      print("Done training")
      
      print("LL")
      ll_bynode<-logLik(bnobj,test_data_imputed,by.node=TRUE)
      ll_all<-sum(ll_bynode[is.finite(ll_bynode)])
    
      print("AIC")
      aic_bynode<-AIC(bnobj,test_data_imputed,by.node=TRUE)
      aic_all<-sum(aic_bynode[is.finite(aic_bynode)])
      
      print("BIC")
      bic_bynode<-BIC(bnobj,test_data_imputed,by.node=TRUE)
      bic_all<-sum(bic_bynode[is.finite(bic_bynode)])
      
      dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                   create_row(
                                     train_dataset_name,
                                     test_dataset_name,
                                     downsamp_n,
                                     80,
                                     20,
                                     'em',
                                     'all',
                                     aic_all,bic_all,ll_all,
                                     percmissing
                                   ));
      for(n in names(ll_bynode)){
        ll_node<-ll_bynode[n]
        aic_node<-aic_bynode[n]
        bic_node<-bic_bynode[n]
        dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                     create_row(
                                       train_dataset_name,
                                       test_dataset_name,
                                       downsamp_n,
                                       80,
                                       20,
                                       'em',
                                       n,
                                       aic_node,
                                       bic_node,
                                       ll_node,
                                       percmissing
                                     ));
        
      }
      print("Done evaluation")
    }
  }
  
  
  #print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}

write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'exp3_metrics.csv',sep=''))

###
#
# Exclusive test
#
###

i<-0
for(col in train_datasets_exclusive){
  i<-i+1
  train_dataset_name<-train_datasets_exclusive[i]
  test_dataset_name<-train_datasets_exclusive[i]
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  #n_downsamps<- 1
  
  for(downsamp_n in (1:n_downsamps)){
    subdir<- 'exclusive'
    train_data<-load_dataset(subdir,'train',downsamp_n,train_dataset_name,80,test_dataset_name,80);
    train_data <-retransform_data(train_data)
    test_data<-load_dataset(subdir,'test',downsamp_n,train_dataset_name,80,test_dataset_name,80);
    test_data <-retransform_data(test_data)
    print(colnames(train_data))
    
    s<-mmhc(train_data)
    print(BIC(s,test_data))
  }
  
  #print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}



###
#
# All Test
#
###
i<-0
for(col in train_datasets_allcolumns){
  i<-i+1
  train_dataset_name<-train_datasets_allcolumns[i]
  test_dataset_name<-test_datasets_allcolumns[i]
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  #n_downsamps<- 1
  
  for(downsamp_n in (1:n_downsamps)){
    subdir<- 'allcolumns'
    train_data<-load_dataset(subdir,'train',downsamp_n,train_dataset_name,100,test_dataset_name,80);
    train_data <-retransform_data(train_data)
    test_data<-load_dataset(subdir,'test',downsamp_n,train_dataset_name,100,test_dataset_name,80);
    test_data <-retransform_data(test_data)
    print(colnames(train_data))
    
    s<-structural.em(train_data,return.all=TRUE)
  }
  
  #print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}

i<-0
for(dname in dataset_names){
  i<-i+1
  print(paste("Dataset name:",dname))
  dataset_metrics_frame<-data.frame()
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  if(dname %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
    n_downsamps <- 1
  }
  
  n_total_rows <- dataset_max_sizes[i]
  for(downsamp_n in (1:n_downsamps)){
    print(paste("Downsample num:",downsamp_n,"of",n_downsamps))
    for(randomsamp_n in (1:N_RANDOM_SAMPLES)){
      print(paste("Random sample num:",randomsamp_n,"of",N_RANDOM_SAMPLES))
      for(n_points in append(seq(N_RANDOM_BASE,n_total_rows,N_RANDOM_INCREMENTS),n_total_rows)){
        #for(percent in samp_percentages){
        #print(paste("N points:",n_points,"of",n_total_rows))
        graph_dataset <-retransform_data(load_dataset(dname,downsamp_n,randomsamp_n,n_points))
        if(dname %in% c('eopn_full','eopn_subset')){
          graph_dataset <- graph_dataset[ , !(names(graph_dataset) %in% c('facet_product_val_intellectual'))]  
        }
        
        #print(colnames(graph_dataset))
        for(algo in algorithms){
          #print(paste("Algo:",algo))
          boot_graphs<-bn.boot(graph_dataset,R=N_BOOTSTRAP_ITERS,algorithm=algo,statistic=function(x) x)
          for(bootstrap_num in (1:N_BOOTSTRAP_ITERS)){
            boot_graph<-boot_graphs[[bootstrap_num]]
            
            # TODO: Can I still extract metrics without this?
            #if(length(undirected.arcs(boot_graph)) > 0){
            #  boot_graph<-cextend(boot_graph)
            #}
            
            
            #TODO: bn.fit? Or bn?
            #TODO: Directed (second-to-last argument) is TRUE.  Make dynamic
            write_graph(boot_graph,dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,length(undirected.arcs(boot_graph)) == 0,FALSE,FALSE)
            #write.dsc('/Users/Matt/Desktop/test.dsc',fitted_boot_graph)
            #saveRDS(fitted_boot_graph,'/Users/Matt/Desktop/test.rds')
            #fitted_boot_graph<-read.dsc('/Users/Matt/Desktop/test.dsc')
            AIC_boot<-AIC(boot_graph,graph_dataset)
            BIC_boot<-BIC(boot_graph,graph_dataset)
            logLik_boot<-logLik(boot_graph,graph_dataset)
            #print(score(boot_graph, graph_dataset, type = "bic-g"))
            #print(dname)
            #print(logLik_boot)
            dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                         create_row(dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,length(undirected.arcs(boot_graph)) == 0,FALSE,FALSE,AIC_boot,BIC_boot,logLik_boot))
            #BF(boot_graph,boot_graph,graph_dataset)
            
            #fitted_boot_graph<-bn.net(bn.fit(boot_graph,graph_dataset))
            #write_graph(fitted_boot_graph,dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,TRUE,TRUE,FALSE)
            #AIC_fitted<-AIC(fitted_boot_graph,graph_dataset)
            #BIC_fitted<-BIC(fitted_boot_graph,graph_dataset)
            #logLik_fitted<-logLik(fitted_boot_graph,graph_dataset)
            #dataset_metrics_frame<-rbind(dataset_metrics_frame,create_row(dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,TRUE,TRUE,FALSE,AIC_fitted,BIC_fitted,logLik_fitted))
            
          }
        }
      }
    }
  }
  
  print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}



dataset_names <- c(
  'eopn_subset','eopn_full',
  'nsf_subset','sal_subset',
  'nsf_full','sal_full',
  'nsf_sal','nsf_eopn','sal_eopn','nsf_sal_eopn'
  #'nsf_sal_subset','nsf_eopn_subset','sal_eopn_subset','nsf_sal_eopn_subset'
  )

dataset_max_sizes <- c(
  1274,1274,
  697,594,
  697,594,
  594*2,697*2,594*2,594*3
)

algorithms <- c(
  #'iamb',
  'mmhc','hc')

#samp_percentages <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
 
load_dataset <- function(dataset_name,downsample_number,randomsample_number,n_points){
  downsamplestring <- paste('_downsample',downsample_number,sep='')
  if(dataset_name %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
    downsamplestring <- '_downsample1'
  }
  randomsamplestring <- paste('_sample',randomsample_number,sep='')
  #percentstring <- paste('_perc',as.integer(percent*100),sep='')
  
  data <- read.csv(paste(DATA_DIRECTORY,dataset_name,downsamplestring,'_n',n_points,'_sample',randomsample_number,'.csv',sep=''),row.names=1)
  #data <- read.csv(paste(DATA_DIRECTORY,dataset_name,'_downsample',downsample_number,'_n',n_points,'_sample',randomsample_number,'.csv',sep=''))
  #data<-read.csv(paste(DATA_DIRECTORY,dataset_name,downsamplestring,percentstring,randomsamplestring,'.csv',sep=''))
  return(data)
}

write_graph <- function(graph,dataset_name,downsample_number,randomsample_number,num_points,bootstrap_number,algorithm,directed,fitted,manual){
  manualstring <- ''
  if(manual){
    manualstring<-'manual_'
  }
  downsamplestring <- paste('_downsample',downsample_number,sep='')
  if(dataset_name %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
    downsamplestring <- ''
  }
  randomsamplestring <- paste('_sample',randomsample_number,sep='')
  nstring <- paste('_n',num_points,sep='')
  #percentstring <- paste('_perc',as.integer(percent*100),sep='')
  
  bootstrapsamplestring <- paste('_bootstrap',bootstrap_number,sep='')
  directedstring<-'_original'
  if(directed){
    directedstring<-'_directed'
  }
  
  
  algorithmstring<-paste('_algo',algorithm,sep='')
  
  fittedstring<-'_unfitted'
  if(fitted){
    fittedstring<-'_fitted'
  }
  
  filename <- paste(GRAPH_DIRECTORY,manualstring,dataset_name,downsamplestring,nstring,randomsamplestring,bootstrapsamplestring,algorithmstring,directedstring,fittedstring,'.rds',sep='')
  #filename <- paste(GRAPH_DIRECTORY,dataset_name,downsamplestring,percentstring,randomsamplestring,bootstrapsamplestring,algorithmstring,directedstring,fittedstring,'.rds',sep='')
  saveRDS(graph,filename)
}

create_row <- function(dataset_name,downsample_number,randomsample_number,num_points,bootstrap_number,algorithm,directed,fitted,manual,AIC_score,BIC_score,logLik_score){
  row <- data.frame(dataset_name=dataset_name,
                    downsample_number=downsample_number,
                    randomsample_number=randomsample_number,
                    bootstrap_number=bootstrap_number,
                    n=num_points,
                    algorithm=algorithm,
                    directed=directed,
                    fitted=fitted,
                    manual=manual,
                    AIC=AIC_score,
                    BIC=BIC_score,
                    logLik=logLik_score
                      )
  return(row)
}






nsf_fulldata_cols<-c(
  'facet_goal_val_amorphous',
  'topic_globalwarming',
  'facet_product_val_intellectual',
  'search_expertise',
  'search_frequency',
  'search_journalism',
  'search_years',
  'topic_familiarity',
  'assignment_experience',
  'task_difficulty',
  'post_rushed',
  
  #'intenttotal_current_id',
  #'intenttotal_current_learn',
  #'intenttotal_current_findaccessobtain',
  #'intenttotal_current_keep',
  #'intenttotal_current_evaluate',
  
  'pages_num_segment',
  'time_dwelltotal_total_content_segment',
  'time_dwelltotal_total_serp_segment',
  'query_length'
)
sal_fulldata_cols <- c(
  'facet_goal_val_amorphous',
  'facet_product_val_intellectual',
  
  'search_expertise',
  'topic_familiarity',
  'task_difficulty',
  
  'search_years',
  'search_frequency',
  
  
  
  'pages_num_segment',
  'time_dwelltotal_total_content_segment',
  'time_dwelltotal_total_serp_segment',
  'query_length'
)

eopn_fulldata_cols <- c(
  'facet_goal_val_amorphous',
  'facet_product_val_intellectual',
  
  'search_expertise',
  'topic_familiarity',
  
  'task_difficulty',
  'post_rushed',
  
  'pages_num_segment',
  'time_dwelltotal_total_content_segment',
  'time_dwelltotal_total_serp_segment',
  'query_length'
)

shared_cols <- c(
  'facet_goal_val_amorphous',
  'facet_product_val_intellectual',
  'pages_num_segment',
  'time_dwelltotal_total_content_segment',
  'time_dwelltotal_total_serp_segment',
  'query_length',
  'topic_familiarity',
  'task_difficulty',
  #'post_difficult',
  'search_expertise'
)



i<-0
for(dname in dataset_names){
  i<-i+1
  print(paste("Dataset name:",dname))
  dataset_metrics_frame<-data.frame()
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  if(dname %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
    n_downsamps <- 1
  }
  
  n_total_rows <- dataset_max_sizes[i]
  for(downsamp_n in (1:n_downsamps)){
    print(paste("Downsample num:",downsamp_n,"of",n_downsamps))
    for(randomsamp_n in (1:N_RANDOM_SAMPLES)){
      print(paste("Random sample num:",randomsamp_n,"of",N_RANDOM_SAMPLES))
      for(n_points in append(seq(N_RANDOM_BASE,n_total_rows,N_RANDOM_INCREMENTS),n_total_rows)){
      #for(percent in samp_percentages){
        #print(paste("N points:",n_points,"of",n_total_rows))
        graph_dataset <-retransform_data(load_dataset(dname,downsamp_n,randomsamp_n,n_points))
        if(dname %in% c('eopn_full','eopn_subset')){
          graph_dataset <- graph_dataset[ , !(names(graph_dataset) %in% c('facet_product_val_intellectual'))]  
        }
        
        #print(colnames(graph_dataset))
        for(algo in algorithms){
          #print(paste("Algo:",algo))
          boot_graphs<-bn.boot(graph_dataset,R=N_BOOTSTRAP_ITERS,algorithm=algo,statistic=function(x) x)
          for(bootstrap_num in (1:N_BOOTSTRAP_ITERS)){
            boot_graph<-boot_graphs[[bootstrap_num]]
            
            # TODO: Can I still extract metrics without this?
            #if(length(undirected.arcs(boot_graph)) > 0){
            #  boot_graph<-cextend(boot_graph)
            #}
            
            
            #TODO: bn.fit? Or bn?
            #TODO: Directed (second-to-last argument) is TRUE.  Make dynamic
            write_graph(boot_graph,dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,length(undirected.arcs(boot_graph)) == 0,FALSE,FALSE)
            #write.dsc('/Users/Matt/Desktop/test.dsc',fitted_boot_graph)
            #saveRDS(fitted_boot_graph,'/Users/Matt/Desktop/test.rds')
            #fitted_boot_graph<-read.dsc('/Users/Matt/Desktop/test.dsc')
            AIC_boot<-AIC(boot_graph,graph_dataset)
            BIC_boot<-BIC(boot_graph,graph_dataset)
            logLik_boot<-logLik(boot_graph,graph_dataset)
            #print(score(boot_graph, graph_dataset, type = "bic-g"))
            #print(dname)
            #print(logLik_boot)
            dataset_metrics_frame<-rbind(dataset_metrics_frame,
                                         create_row(dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,length(undirected.arcs(boot_graph)) == 0,FALSE,FALSE,AIC_boot,BIC_boot,logLik_boot))
            #BF(boot_graph,boot_graph,graph_dataset)
            
            #fitted_boot_graph<-bn.net(bn.fit(boot_graph,graph_dataset))
            #write_graph(fitted_boot_graph,dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,TRUE,TRUE,FALSE)
            #AIC_fitted<-AIC(fitted_boot_graph,graph_dataset)
            #BIC_fitted<-BIC(fitted_boot_graph,graph_dataset)
            #logLik_fitted<-logLik(fitted_boot_graph,graph_dataset)
            #dataset_metrics_frame<-rbind(dataset_metrics_frame,create_row(dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,TRUE,TRUE,FALSE,AIC_fitted,BIC_fitted,logLik_fitted))
            
          }
        }
      }
    }
  }
  
  print(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}



##########
#
# PART TWO: CURATED GRAPH ANALYSIS
#
##########


nsf_full_manualedges_directed <-
# NSF FULL
# Directed
list(
c('facet_goal_val_amorphous','pages_num_segment'),
c('facet_goal_val_amorphous','time_dwelltotal_total_content_segment'),
c('facet_goal_val_amorphous','time_dwelltotal_total_serp_segment'),
c('facet_goal_val_amorphous','query_length'),

c('facet_product_val_intellectual','pages_num_segment'),
c('facet_product_val_intellectual','time_dwelltotal_total_content_segment'),
c('facet_product_val_intellectual','time_dwelltotal_total_serp_segment'),
c('facet_product_val_intellectual','query_length'),


c('topic_familiarity','task_difficulty'),
c('search_expertise','task_difficulty'),
c('search_frequency','task_difficulty'),
c('search_years','task_difficulty'),
c('assignment_experience','task_difficulty'),
c('post_rushed','task_difficulty'),

c('facet_goal_val_amorphous','task_difficulty'),
c('facet_product_val_intellectual','task_difficulty'),


c('topic_globalwarming','topic_familiarity'),

c('topic_globalwarming','pages_num_segment'),
c('topic_globalwarming','time_dwelltotal_total_content_segment'),
c('topic_globalwarming','time_dwelltotal_total_serp_segment'),
c('topic_globalwarming','query_length'),

c('topic_familiarity','pages_num_segment'),
c('topic_familiarity','time_dwelltotal_total_content_segment'),
c('topic_familiarity','time_dwelltotal_total_serp_segment'),
c('topic_familiarity','query_length'),

c('task_difficulty','pages_num_segment'),
c('task_difficulty','time_dwelltotal_total_content_segment'),
c('task_difficulty','time_dwelltotal_total_serp_segment'),
c('task_difficulty','query_length'),


c('post_rushed','pages_num_segment'),
c('post_rushed','time_dwelltotal_total_content_segment'),
c('post_rushed','time_dwelltotal_total_serp_segment'),
c('post_rushed','query_length'),

c('search_journalism','assignment_experience'),
c('assignment_experience','pages_num_segment'),
c('assignment_experience','time_dwelltotal_total_content_segment'),
c('assignment_experience','time_dwelltotal_total_serp_segment'),
c('assignment_experience','query_length')
);
# Undirected
#('facet_goal_val_amorphous','facet_product_val_intellectual')
#('search_years','search_frequency')




#TODO: Shared
# Directed
shared_manualedges_directed <- list(
c('facet_goal_val_amorphous','pages_num_segment'),
c('facet_goal_val_amorphous','time_dwelltotal_total_content_segment'),
c('facet_goal_val_amorphous','time_dwelltotal_total_serp_segment'),
c('facet_goal_val_amorphous','query_length'),

c('facet_product_val_intellectual','pages_num_segment'),
c('facet_product_val_intellectual','time_dwelltotal_total_content_segment'),
c('facet_product_val_intellectual','time_dwelltotal_total_serp_segment'),
c('facet_product_val_intellectual','query_length'),

c('topic_familiarity','task_difficulty'),
c('search_expertise','task_difficulty'),

c('facet_goal_val_amorphous','task_difficulty'),
c('facet_product_val_intellectual','task_difficulty'),

c('topic_familiarity','pages_num_segment'),
c('topic_familiarity','time_dwelltotal_total_content_segment'),
c('topic_familiarity','time_dwelltotal_total_serp_segment'),
c('topic_familiarity','query_length'),

c('task_difficulty','pages_num_segment'),
c('task_difficulty','time_dwelltotal_total_content_segment'),
c('task_difficulty','time_dwelltotal_total_serp_segment'),
c('task_difficulty','query_length')

# Undirected
#('facet_goal_val_amorphous','facet_product_val_intellectual')
);




#TODO: SAL Full
sal_full_manualedges_directed <- list(
# Directed
c('facet_goal_val_amorphous','pages_num_segment'),
c('facet_goal_val_amorphous','time_dwelltotal_total_content_segment'),
c('facet_goal_val_amorphous','time_dwelltotal_total_serp_segment'),
c('facet_goal_val_amorphous','query_length'),

c('facet_product_val_intellectual','pages_num_segment'),
c('facet_product_val_intellectual','time_dwelltotal_total_content_segment'),
c('facet_product_val_intellectual','time_dwelltotal_total_serp_segment'),
c('facet_product_val_intellectual','query_length'),


c('topic_familiarity','task_difficulty'),
c('search_expertise','task_difficulty'),
c('search_frequency','task_difficulty'),
c('search_years','task_difficulty'),
#c('post_rushed','task_difficulty'),

c('facet_goal_val_amorphous','task_difficulty'),
c('facet_product_val_intellectual','task_difficulty'),


#c('topic_globalwarming','topic_familiarity'),

#c('topic_globalwarming','pages_num_segment'),
#c('topic_globalwarming','time_dwelltotal_total_content_segment'),
#c('topic_globalwarming','time_dwelltotal_total_serp_segment'),
#c('topic_globalwarming','query_length'),

c('topic_familiarity','pages_num_segment'),
c('topic_familiarity','time_dwelltotal_total_content_segment'),
c('topic_familiarity','time_dwelltotal_total_serp_segment'),
c('topic_familiarity','query_length'),

c('task_difficulty','pages_num_segment'),
c('task_difficulty','time_dwelltotal_total_content_segment'),
c('task_difficulty','time_dwelltotal_total_serp_segment'),
c('task_difficulty','query_length')


#c('post_rushed','pages_num_segment'),
#c('post_rushed','time_dwelltotal_total_content_segment'),
#c('post_rushed','time_dwelltotal_total_serp_segment'),
#c('post_rushed','query_length')
);

# Undirected
#('facet_goal_val_amorphous','facet_product_val_intellectual')
#('search_years','search_frequency')


#TODO: EOP Full
# Directed
eopn_full_manualedges_directed <- list(
c('facet_goal_val_amorphous','pages_num_segment'),
c('facet_goal_val_amorphous','time_dwelltotal_total_content_segment'),
c('facet_goal_val_amorphous','time_dwelltotal_total_serp_segment'),
c('facet_goal_val_amorphous','query_length'),

c('facet_product_val_intellectual','pages_num_segment'),
c('facet_product_val_intellectual','time_dwelltotal_total_content_segment'),
c('facet_product_val_intellectual','time_dwelltotal_total_serp_segment'),
c('facet_product_val_intellectual','query_length'),


c('topic_familiarity','task_difficulty'),
c('search_expertise','task_difficulty'),
#c('assignment_experience','task_difficulty'),
c('post_rushed','task_difficulty'),

c('facet_goal_val_amorphous','task_difficulty'),
c('facet_product_val_intellectual','task_difficulty'),


#c('topic_globalwarming','topic_familiarity'),

#c('topic_globalwarming','pages_num_segment'),
#c('topic_globalwarming','time_dwelltotal_total_content_segment'),
#c('topic_globalwarming','time_dwelltotal_total_serp_segment'),
#c('topic_globalwarming','query_length'),

c('topic_familiarity','pages_num_segment'),
c('topic_familiarity','time_dwelltotal_total_content_segment'),
c('topic_familiarity','time_dwelltotal_total_serp_segment'),
c('topic_familiarity','query_length'),

c('task_difficulty','pages_num_segment'),
c('task_difficulty','time_dwelltotal_total_content_segment'),
c('task_difficulty','time_dwelltotal_total_serp_segment'),
c('task_difficulty','query_length'),


c('post_rushed','pages_num_segment'),
c('post_rushed','time_dwelltotal_total_content_segment'),
c('post_rushed','time_dwelltotal_total_serp_segment'),
c('post_rushed','query_length')

# Undirected
#('facet_goal_val_amorphous','facet_product_val_intellectual')
);


dataset_names <- c(
  'nsf_full','sal_full','eopn_full',
  'nsf_subset','sal_subset','eopn_subset',
  'nsf_sal','nsf_eopn','sal_eopn','nsf_sal_eopn'
  #'nsf_sal_subset','nsf_eopn_subset','sal_eopn_subset','nsf_sal_eopn_subset'
)

dataset_max_sizes <- c(
  697,594,1274,
  697,594,1274,
  594*2,697*2,594*2,594*3
)


manual_edges <- list(
  nsf_full_manualedges_directed,sal_full_manualedges_directed,eopn_full_manualedges_directed,
  shared_manualedges_directed,shared_manualedges_directed,shared_manualedges_directed,
  shared_manualedges_directed,shared_manualedges_directed,shared_manualedges_directed,shared_manualedges_directed
)

construct_graph <- function(data,edges,name){
  graph <- empty.graph(colnames(data))
  edgelist <- c()
  #print("EDGES")
  #print(edges)
  #print("END EDGES")
  for(edgenum in 1:length(edges)){
    
    
    if(name %in% c('eopn_full','eopn_subset')){
      #print(edges[[edgenum]][1])
      if(edges[[edgenum]][1]=='facet_product_val_intellectual'){
        next;
      }
      if(edges[[edgenum]][2]=='facet_product_val_intellectual'){
        next;
      }
    }
    edgelist<- append(edgelist,edges[[edgenum]][1])
    edgelist<- append(edgelist,edges[[edgenum]][2])
  }
  
  arc.set = matrix(edgelist,
                   ncol = 2, byrow = TRUE,
                   dimnames = list(NULL, c("from", "to")));
  arcs(graph) = arc.set
  return(graph)
}

print("PART TWO")
i<-0
for(dname in dataset_names){
  i<-i+1
  print(paste("Dataset name:",dname))
  dataset_metrics_frame<-data.frame()
  n_downsamps<- N_DOWNSAMPLED_SAMPLES
  if(dname %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
    n_downsamps <- 1
  }
  
  n_total_rows <- dataset_max_sizes[i]
  for(downsamp_n in (1:n_downsamps)){
    print(paste("Downsample num:",downsamp_n,"of",n_downsamps))
    for(randomsamp_n in (1:N_RANDOM_SAMPLES)){
      print(paste("Random sample num:",randomsamp_n,"of",N_RANDOM_SAMPLES))
      for(n_points in append(seq(N_RANDOM_BASE,n_total_rows,N_RANDOM_INCREMENTS),n_total_rows)){
        #for(percent in samp_percentages){
        #print(paste("N points:",n_points,"of",n_total_rows))
        graph_dataset <-retransform_data(load_dataset(dname,downsamp_n,randomsamp_n,n_points))
        if(dname %in% c('eopn_full','eopn_subset')){
          graph_dataset<- graph_dataset[ , !(names(graph_dataset) %in% c('facet_product_val_intellectual'))]  
        }
        
        #boot_graph<-construct_graph(graph_dataset,manual_edges[i])
        #fitted_boot_graph<-bn.net(bn.fit(boot_graph,graph_dataset))
        boot_graph<-construct_graph(graph_dataset,manual_edges[[i]],dname)
        boot_graph<-bn.net(bn.fit(boot_graph,graph_dataset))
        #TODO: bn.fit? Or bn?
        
        #TODO: Directed (second-to-last argument) is TRUE.  Make dynamic
        #write_graph(fitted_boot_graph,dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,'manual',TRUE,TRUE,TRUE)
        write_graph(boot_graph,dname,downsamp_n,randomsamp_n,n_points,1,'manual',length(undirected.arcs(boot_graph)) == 0,FALSE,TRUE)
        
        AIC_boot<-AIC(boot_graph,graph_dataset)
        BIC_boot<-BIC(boot_graph,graph_dataset)
        logLik_boot<-logLik(boot_graph,graph_dataset)
        dataset_metrics_frame<-rbind(dataset_metrics_frame,create_row(dname,downsamp_n,randomsamp_n,n_points,1,'manual',length(undirected.arcs(boot_graph)) == 0,FALSE,TRUE,AIC_boot,BIC_boot,logLik_boot))
        
        
        #AIC_fitted<-AIC(fitted_boot_graph,graph_dataset)
        #BIC_fitted<-BIC(fitted_boot_graph,graph_dataset)
        #logLik_fitted<-logLik(fitted_boot_graph,graph_dataset)
        #dataset_metrics_frame<-rbind(dataset_metrics_frame,create_row(dname,downsamp_n,randomsamp_n,n_points,bootstrap_num,algo,TRUE,TRUE,TRUE,AIC_fitted,BIC_fitted,logLik_fitted))
        
        
      }
    }
  }
  
  write.csv(dataset_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'manual_',dname,'_metrics.csv',sep=''))
  #close(paste(EVAL_METRICS_DIRECTORY,dname,'_metrics.csv',sep=''))
}




#'facet_goal_val_amorphous',
#'topic_globalwarming',
#'facet_product_val_intellectual',
#'search_expertise',
#'search_frequency',
#'search_journalism',
#'search_years',
#'topic_familiarity',
#'assignment_experience',
#'task_difficulty',
#'post_rushed',


#'pages_num_segment',
#'time_dwelltotal_total_content_segment',
#'time_dwelltotal_total_serp_segment',
#'query_length'


##########
#
# END
#
##########






nsf_fulldata_cols_notaskbehavior<-c(
  'topic_globalwarming',
  'search_expertise',
  'search_frequency',
  'search_journalism',
  'search_years',
  'topic_familiarity',
  'assignment_experience',
  'task_difficulty',
  'post_rushed'
)
sal_fulldata_cols_notaskbehavior <- c(
  
  'search_expertise',
  'topic_familiarity',
  'task_difficulty',
  
  'search_years',
  'search_frequency'
)

eopn_fulldata_cols_notaskbehavior <- c(
  
  'search_expertise',
  'topic_familiarity',
  
  'task_difficulty',
  'post_rushed'
)

shared_cols_notaskbehavior<-c(
  'topic_familiarity',
  'task_difficulty',
  #'post_difficult',
  'search_expertise'
)


dataset_names <- c(
  'eopn_subset','eopn_full',
  'nsf_subset','sal_subset',
  'nsf_full','sal_full',
  'nsf_sal','nsf_eopn','sal_eopn','nsf_sal_eopn'
  #'nsf_sal_subset','nsf_eopn_subset','sal_eopn_subset','nsf_sal_eopn_subset'
)

dataset_max_sizes <- c(
  1274,1274,
  697,594,
  697,594,
  594*2,697*2,594*2,594*3
)
dataset_cols <- list(
  shared_cols_notaskbehavior,eopn_fulldata_cols_notaskbehavior,
  shared_cols_notaskbehavior,shared_cols_notaskbehavior,
  nsf_fulldata_cols_notaskbehavior,sal_fulldata_cols_notaskbehavior,
  shared_cols_notaskbehavior,shared_cols_notaskbehavior,shared_cols_notaskbehavior,shared_cols_notaskbehavior
)

df_full<-data.frame()
for (i in 1:length(dataset_names)){
  datasetname<-dataset_names[i]
  n_points<-dataset_max_sizes[i]
  datacols<-dataset_cols[[i]]
  df_data<-data.frame()
  print(paste('Data:',datasetname))
  for(downsample_num in 1:N_DOWNSAMPLED_SAMPLES){
    print(paste("Downsample num",downsample_num))
    for(randomsample_num in 1:N_RANDOM_SAMPLES){
      #print(paste("Randomsample num",randomsample_num))
      data<- retransform_data(load_dataset(datasetname,downsample_num,randomsample_num,n_points))
      for(ncols in 0:length(datacols)){
        cols_sample<-sample(datacols,ncols)
        data_train<-data[ , !(names(data) %in% cols_sample)]
        nvars<-ncol(data_train)
        #print(paste("nvars",nvars))
        for(algo in algorithms){
          boot_graph<-NULL
          if(algo=='mmhc'){
            boot_graph<-mmhc(data_train)
          }else if(algo=='hc'){
            boot_graph<-hc(data_train)
          }
          #boot_graphs<-bn.boot(data_train,R=1,algorithm=algo,statistic=function(x) x)
          #boot_graph<-boot_graphs[[1]]
          logLik_boot<-logLik(boot_graph,data_train)
          BIC_boot<-BIC(boot_graph,data_train)
          row<-data.frame(
            downsample_number=downsample_num,
            dataset=datasetname,
            n=n_points,
            randomsample_number=randomsample_num,
            nvars=nvars,
            algorithm=algo,
            logLik=logLik_boot,
            BIC=BIC_boot
          )
          df_data<-rbind(df_data,row)
        }
      }
      
    }
    print(nrow(df_data))
  }
  
  df_full<-rbind(df_full,df_data)
  print("NROW")
  print(nrow(df_full))
  
}

write.csv(df_full,paste(EVAL_METRICS_DIRECTORY,'loglik_scores.csv',sep=''))


















df_full<-data.frame()
for (i in 1:length(dataset_names)){
  datasetname<-dataset_names[i]
  n_points<-dataset_max_sizes[i]
  datacols<-dataset_cols[[i]]
  df_data<-data.frame()
  print(paste('Data:',datasetname))
  for(downsample_num in 1:5){
    print(paste("Downsample num",downsample_num))
    for(randomsample_num in 1:5){
      #print(paste("Randomsample num",randomsample_num))
      data<- retransform_data(load_dataset(datasetname,downsample_num,randomsample_num,n_points))
      for(ncols in 0:length(datacols)){
        cols_sample<-sample(datacols,ncols)
        #print("total cols")
        #print(colnames(data))
        #print("sample cols")
        #print(cols_sample)
        #print("data cols")
        #print(datacols)
        data_train<-data[ , !(names(data) %in% cols_sample)]
        
        if(datasetname=='eopn_full' || datasetname=='eopn_subset'){
          data_train<-data_train[ , !(names(data_train) %in% c('facet_product_val_intellectual'))]
        }
        
        nvars<-ncol(data_train)
        #print(paste("nvars",nvars))
        for(algo in algorithms){
          boot_graphs<-NULL
          if(algo=='mmhc'){
            boot_graphs<-bn.cv(data_train,'mmhc',k=5)
          }else if(algo=='hc'){
            boot_graphs<-bn.cv(data_train,'hc',method='k-fold',k=5)
          }
          for(k in 1:5){
            boot_graph <- boot_graphs[[k]]
            #logLik_boot<-logLik(boot_graph,data_train)
            #BIC_boot<-BIC(boot_graph,data_train)
            row<-data.frame(
              downsample_number=downsample_num,
              dataset=datasetname,
              n=n_points,
              randomsample_number=randomsample_num,
              nvars=nvars,
              algorithm=algo,
              logLik_loss=boot_graph$loss,
              k=k
            )
            df_data<-rbind(df_data,row)
          }
        }
      }
      
    }
    print(nrow(df_data))
  }
  
  df_full<-rbind(df_full,df_data)
  print("NROW")
  print(nrow(df_full))
  
}

write.csv(df_full,paste(EVAL_METRICS_DIRECTORY,'loss_scores.csv',sep=''))














df_full<-data.frame()
for (i in 1:length(dataset_names)){
  datasetname<-dataset_names[i]
  #n_points<-dataset_max_sizes[i]
  datacols<-dataset_cols[[i]]
  df_data<-data.frame()
  print(paste('Data:',datasetname))
  for(downsample_num in 1:3){
    print(paste("Downsample num",downsample_num))
    for(randomsample_num in 1:3){
      #print(paste("Randomsample num",randomsample_num))
      data<- retransform_data(load_dataset(datasetname,downsample_num,randomsample_num,dataset_max_sizes[i]))
      for(n_points in seq(100,dataset_max_sizes[i],100)){
        for(ncols in 0:length(datacols)){
          cols_sample<-sample(datacols,ncols)
          #print("total cols")
          #print(colnames(data))
          #print("sample cols")
          #print(cols_sample)
          #print("data cols")
          #print(datacols)
          data_train<-data[ , !(names(data) %in% cols_sample)]
          
          if(datasetname=='eopn_full' || datasetname=='eopn_subset'){
            data_train<-data_train[ , !(names(data_train) %in% c('facet_product_val_intellectual'))]
          }
          
          nvars<-ncol(data_train)
          #print(paste("nvars",nvars))
          for(algo in algorithms){
            boot_graphs<-NULL
            if(algo=='mmhc'){
              boot_graphs<-bn.cv(data_train,'mmhc',k=5)
            }else if(algo=='hc'){
              boot_graphs<-bn.cv(data_train,'hc',method='k-fold',k=5)
            }
            for(k in 1:5){
              boot_graph <- boot_graphs[[k]]
              #logLik_boot<-logLik(boot_graph,data_train)
              #BIC_boot<-BIC(boot_graph,data_train)
              row<-data.frame(
                downsample_number=downsample_num,
                dataset=datasetname,
                n=n_points,
                randomsample_number=randomsample_num,
                nvars=nvars,
                algorithm=algo,
                logLik_loss=boot_graph$loss,
                k=k
              )
              df_data<-rbind(df_data,row)
            }
          }
        }
        
      }
      
      
    }
    print(nrow(df_data))
  }
  
  df_full<-rbind(df_full,df_data)
  print("NROW")
  print(nrow(df_full))
  
}

write.csv(df_full,paste(EVAL_METRICS_DIRECTORY,'loss_scores_byn.csv',sep=''))

