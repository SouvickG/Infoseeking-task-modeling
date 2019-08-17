library(bnlearn)
library(Rgraphviz)


N_DOWNSAMPLED_SAMPLES <- 20
N_RANDOM_SAMPLES <- 20
#N_DOWNSAMPLED_SAMPLES <- 50
#N_RANDOM_SAMPLES <- 50
N_RANDOM_BASE <- 100
N_RANDOM_INCREMENTS <- 100
N_BOOTSTRAP_ITERS <- 20
N_TESTS <- 20
DATA_DIRECTORY <- '/Users/Matt/Desktop/Experiment/data/interim/graph_data_samples/'
GRAPH_DIRECTORY <- '/Users/Matt/Desktop/Experiment/data/interim/trained_structure_graphs/'
EVAL_METRICS_DIRECTORY <- '/Users/Matt/Desktop/Experiment/data/interim/structure_graph_metrics/'

#TODO: Full parameters
dataset_names <- c(
  'nsf_subset','sal_subset','eopn_subset',
  'nsf_full','sal_full','eopn_full',
  'nsf_sal','nsf_eopn','sal_eopn','nsf_sal_eopn'
  #'nsf_sal_subset','nsf_eopn_subset','sal_eopn_subset','nsf_sal_eopn_subset'
)

dataset_max_sizes <- c(
  697,594,1274,
  697,594,1274,
  594*2,697*2,594*2,594*3
)


DATASET_NAMES <- c(
  'nsf_subset','sal_subset','eopn_subset',
  'nsf_full','sal_full','eopn_full',
  'nsf_sal','nsf_eopn','sal_eopn','nsf_sal_eopn'
  #'nsf_sal_subset','nsf_eopn_subset','sal_eopn_subset','nsf_sal_eopn_subset'
)

DATASET_MAX_SIZES <- c(
  697,594,1274,
  697,594,1274,
  594*2,697*2,594*2,594*3
)

algorithms <- c(
  #'iamb',
  'mmhc','hc')

load_dataset <-function(dataset_name,downsample_number,randomsample_number,n_points){
  downsamplestring <- paste('_downsample',downsample_number,sep='')
  if(dataset_name %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
    downsamplestring <- '_downsample1'
  }
  randomsamplestring <- paste('_sample',randomsample_number,sep='')
  #percentstring <- paste('_perc',as.integer(percent*100),sep='')
  
  data <- read.csv(paste(DATA_DIRECTORY,dataset_name,downsamplestring,'_n',n_points,'_sample',randomsample_number,'.csv',sep=''),row.names=1)
  print(paste("Directory:",DATA_DIRECTORY,dataset_name,downsamplestring,'_n',n_points,'_sample',randomsample_number,'.csv',sep=''))
  #data <- read.csv(paste(DATA_DIRECTORY,dataset_name,'_downsample',downsample_number,'_n',n_points,'_sample',randomsample_number,'.csv',sep=''))
  #data<-read.csv(paste(DATA_DIRECTORY,dataset_name,downsamplestring,percentstring,randomsamplestring,'.csv',sep=''))
  return(data)
}


get_max_points <- function(datasetname){
  for(i in 1:length(DATASET_MAX_SIZES)){
    if(datasetname==DATASET_NAMES[i]){
      return(DATASET_MAX_SIZES[i])
    }
  }
}


read_graph <- function(dataset_name,downsample_number,randomsample_number,num_points,bootstrap_number,algorithm,directed,fitted,manual){
  manualstring <- ''
  if(manual){
    manualstring<-'manual_'
  }
  
  if(algorithm=='manual'){
    bootstrap_number <- 1
  }
  
  nstring <- paste('_n',num_points,sep='')
  downsamplestring <- paste('_downsample',downsample_number,sep='')
  if(dataset_name %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
    downsamplestring <- ''
  }
  randomsamplestring <- paste('_sample',randomsample_number,sep='')
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
  
  #print("Graph Directory:")
  #print(paste(GRAPH_DIRECTORY,manualstring,dataset_name,downsamplestring,nstring,randomsamplestring,bootstrapsamplestring,algorithmstring,directedstring,fittedstring,'.rds',sep=''))
  filename <- paste(GRAPH_DIRECTORY,manualstring,dataset_name,downsamplestring,nstring,randomsamplestring,bootstrapsamplestring,algorithmstring,directedstring,fittedstring,'.rds',sep='')
  #filename <- paste(GRAPH_DIRECTORY,dataset_name,downsamplestring,percentstring,randomsamplestring,bootstrapsamplestring,algorithmstring,directedstring,fittedstring,'.rds',sep='')
  return(readRDS(filename))
}

create_pairmetric_row <- function(
  dataset_name1,downsample_number1,randomsample_number1,n_points1,bootstrap_number1,algorithm1,directed1,fitted1,manual1,n_arcs1,
  dataset_name2,downsample_number2,randomsample_number2,n_points2,bootstrap_number2,algorithm2,directed2,fitted2,manual2,n_arcs2,
  precision,recall,
  BF_score,BF_max,BF_log,BF_log_max,shd_score,
  #hamming_score,
  test_number){
  row <- data.frame(
    dataset_name1=dataset_name1,
                    downsample_number1=downsample_number1,
                    randomsample_number1=randomsample_number1,
                    bootstrap_number1=bootstrap_number1,
                    n1=n_points1,
                    algorithm1=algorithm1,
                    directed1=directed1,
                    fitted1=fitted1,
                    manual1=manual1,
                    n_arcs1=n_arcs1,
    
    dataset_name2=dataset_name2,
    downsample_number2=downsample_number2,
    randomsample_number2=randomsample_number2,
    bootstrap_number2=bootstrap_number2,
    n2=n_points2,
    algorithm2=algorithm2,
    directed2=directed2,
    fitted2=fitted2,
    manual2=manual2,
    n_arcs2=n_arcs2,
    
                    test_number=test_number,
    precision=precision,
    recall=recall,
                    BF=BF_score,
                    BF_max=BF_max,
                    BF_log=BF_log,
                    BF_log_max=BF_log_max,
                    SHD=shd_score
    #,
                    #Hamming=hamming_score
    
                      )
  return(row)
}


retransform_data <- function(data){
  data[] <- lapply(data, function(x) {
    if(is.factor(x) || is.integer(x)) as.numeric(as.character(x)) else x
  })
  return(data)
}


BF_score_helper<-function(datasetname1,datasetname2,graph1,graph2,analysis_dataset,scoretype){
  if((datasetname1 %in% c('eopn_full','eopn_subset')) && !(datasetname2 %in% c('eopn_full','eopn_subset'))){
    return(0)
  }
  else if((datasetname2 %in% c('eopn_full','eopn_subset')) && !(datasetname1 %in% c('eopn_full','eopn_subset'))){
    return(0)
  }
  
  if(scoretype=='normal'){
    return(BF(graph1,graph2,analysis_dataset,log=FALSE))
  }
  if(scoretype=='max'){
    return(max(BF(graph1,graph2,analysis_dataset,log=FALSE),BF(graph2,graph1,analysis_dataset,log=FALSE)))
  }
  if(scoretype=='log'){
    return(BF(graph1,graph2,analysis_dataset,log=TRUE))
  }
  if(scoretype=='logmax'){
    return(max(BF(graph1,graph2,analysis_dataset,log=TRUE),BF(graph2,graph1,analysis_dataset,log=TRUE)))  
  }
}

shd_score_helper<-function(datasetname1,datasetname2,graph1,graph2){
  constant <- 0
  shd_manual<- 0
  otherdataset<-''
  #if(datasetname1 =='eopn_full' && !(datasetname2 %in% c('eopn_full','eopn_subset'))){
  #  otherdataset<-datasetname2
  #  if(grepl('_full',datasetname1)){
  #    if(datasetname<-){
  #      
  #    }
  #  }else{
  #    constant <- 1
  #  }
  #}
  #if(datasetname2 =='eopn_subset' && !(datasetname1 %in% c('eopn_full','eopn_subset'))){
  #  otherdataset<-datasetname1
  #}
#  
#  if(datasetname1 =='eopn_subset' && !(datasetname2 %in% c('eopn_full','eopn_subset'))){
#    otherdataset<-datasetname2
#  }
#  if(datasetname2 =='eopn_full' && !(datasetname1 %in% c('eopn_full','eopn_subset'))){
#    otherdataset<-datasetname1
#  }
##  
#  if(otherdataset!=''){
#    if(grepl('_full',otherdataset)){
#      if(otherdataset=='nsf_full'){
##        nsf<-15-9+1
#      }
#      if(otherdataset=='sal_full'){
#        sal<-11-9+1
#      }
#    }
#  }
  
  
 # if((datasetname1 %in% c('eopn_full','eopn_subset')) && !(datasetname2 %in% c('eopn_full','eopn_subset'))){
#    shd_manual<-0
 # }
#  if((datasetname2 %in% c('eopn_full','eopn_subset')) && !(datasetname1 %in% c('eopn_full','eopn_subset'))){
#    shd_manual<-0
#  }
#  
  
  n_arcs1<-narcs(graph1)
  n_arcs2<-narcs(graph2)
  arcs1<-arcs(graph1)
  arcs2<-arcs(graph2)
  nrows1<-nrow(arcs1)
  nrows2<-nrow(arcs2)
  #print(arcs1)
  #print(arcs2)
  overlap<-sum(duplicated(rbind(arcs2,arcs1)))
  
  if((datasetname1 %in% c('eopn_full','eopn_subset')) && !(datasetname2 %in% c('eopn_full','eopn_subset'))){
      shd1<-nrows1-(sum(duplicated(rbind(arcs2,arcs1))))
      shd2<-nrows2-(sum(duplicated(rbind(arcs1,arcs2))))
      return(shd1+shd2)
   }
   else if((datasetname2 %in% c('eopn_full','eopn_subset')) && !(datasetname1 %in% c('eopn_full','eopn_subset'))){
     shd1<-nrows1-(sum(duplicated(rbind(arcs2,arcs1))))
     shd2<-nrows2-(sum(duplicated(rbind(arcs1,arcs2))))
     return(shd1+shd2)
   }
  else{
    return(shd(graph1,graph2));  
  }
  #if(constant==0){
  #  return(shd(graph1,graph2));  
  #}else{
  #  return(constant+shd_manual);  
  #}
  
}
create_pairmetric_row_helper <-function(datasetname1,datasetname2,analysisdataname,algo1,algo2,directed,fitted,manual1,manual2,testnum){
  if(datasetname1!=datasetname2){
    downsamplenum1<-sample(N_DOWNSAMPLED_SAMPLES,1)
    if(datasetname1 %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
      downsamplenum1 <- 1
    }
    downsamplenum2<-sample(N_DOWNSAMPLED_SAMPLES,1)
    if(datasetname2 %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
      downsamplenum2 <- 1
    }
    downsamplenum3<-sample(N_DOWNSAMPLED_SAMPLES,1)
    if(analysisdataname %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
      downsamplenum3 <- 1
    }
    
    randomsamplenum1<-sample(N_RANDOM_SAMPLES,1)
    randomsamplenum2<-sample(N_RANDOM_SAMPLES,1)
    randomsamplenum3<-sample(N_RANDOM_SAMPLES,1)
    
    bootstramsamplenum1<-sample(N_BOOTSTRAP_ITERS,1)
    bootstramsamplenum2<-sample(N_BOOTSTRAP_ITERS,1) 
    bootstramsamplenum3<-sample(N_BOOTSTRAP_ITERS,1) 
  }else{
    s<-sample(N_DOWNSAMPLED_SAMPLES,3)
    downsamplenum1<-s[1]
    downsamplenum2<-s[2]
    downsamplenum3<-s[3]
    if(datasetname1 %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
      downsamplenum1<-1
      downsamplenum2<-1
    }
    
    if(analysisdataname %in% c('nsf_full','sal_full','eopn_full','nsf_subset','sal_subset','eopn_subset')){
      downsamplenum3<-1
    }
    
    
    s<-sample(N_RANDOM_SAMPLES,3)
    randomsamplenum1<-s[1]
    randomsamplenum2<-s[2]
    randomsamplenum3<-s[3]
    
    s<-sample(N_BOOTSTRAP_ITERS,3)
    bootstramsamplenum1<-s[1]
    bootstramsamplenum2<-s[2]
    bootstramsamplenum3<-s[3]
    
  }
  
  
  
  n_points1 <- get_max_points(datasetname1)
  n_points2 <- get_max_points(datasetname2)
  n_points3 <- get_max_points(analysisdataname)
  
  analysis_dataset<- retransform_data(load_dataset(analysisdataname,downsamplenum3,randomsamplenum3,n_points3))
  if((datasetname1 %in% c('eopn_full','eopn_subset')) || (datasetname2 %in% c('eopn_full','eopn_subset'))){
    analysis_dataset<- analysis_dataset[ , !(names(analysis_dataset) %in% c('facet_product_val_intellectual'))]  
  }
  #print("DATA INFO")
  #print(colnames(analysis_dataset))
  #print(analysisdataname)
  graph1<-read_graph(datasetname1,downsamplenum1,randomsamplenum1,n_points1,bootstramsamplenum1,algo1,directed,fitted,manual1)
  
  graph2<-read_graph(datasetname2,downsamplenum2,randomsamplenum2,n_points2,bootstramsamplenum2,algo2,directed,fitted,manual2)
  #print(graph1)
  #print(graph2)
  #print(typeof(graph1))
  #print(typeof(graph2))
  #print(typeof(analysis_dataset))
  n_arcs1<-narcs(graph1)
  n_arcs2<-narcs(graph2)
  arcs1<-arcs(graph1)
  arcs2<-arcs(graph2)
  #print(arcs1)
  #print(arcs2)
  overlap<-sum(duplicated(rbind(arcs2,arcs1)))
  prec<-overlap/nrow(arcs1)
  rec<-overlap/nrow(arcs2)
  BF_score<-BF_score_helper(datasetname1,datasetname2,graph1,graph2,analysis_dataset,'normal')
  BF_log<-BF_score_helper(datasetname1,datasetname2,graph1,graph2,analysis_dataset,'log')
  BF_max<-BF_score_helper(datasetname1,datasetname2,graph1,graph2,analysis_dataset,'max')
  BF_log_max<-BF_score_helper(datasetname1,datasetname2,graph1,graph2,analysis_dataset,'logmax')
  
  shd_score<-shd_score_helper(datasetname1,datasetname2,graph1,graph2)
  #hamming_score<-hamming(graph1,graph2)
  
  return (create_pairmetric_row(
    datasetname1,downsamplenum1,randomsamplenum1,n_points1,bootstramsamplenum1,algo1,directed,fitted,manual1,n_arcs1,
    datasetname2,downsamplenum2,randomsamplenum2,n_points2,bootstramsamplenum2,algo2,directed,fitted,manual2,n_arcs2,
    prec,rec,
    BF_score,BF_max,BF_log,BF_log_max,shd_score,
    #hamming_score,
    testnum
  ));
  
}


##########
#
# Part One: Manual vs. automatic
#
##########

dataset_names_2 <- c('nsf_subset','sal_subset','eopn_subset',
                     'nsf_full','sal_full','eopn_full')
dataset_names_1 <- c('nsf_subset','sal_subset','eopn_subset',
                     'nsf_full','sal_full','eopn_full')
analysisdataset_names <- c('nsf_subset','sal_subset','eopn_subset',
                     'nsf_full','sal_full','eopn_full')
manuals_2 <- c(TRUE,TRUE,TRUE,TRUE,TRUE,TRUE)
manuals_1 <- c(FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)
for(dataset_index in 1:length(dataset_names_1)){
  datasetname1<-dataset_names_1[dataset_index]
  datasetname2<-dataset_names_2[dataset_index]
  print(paste("Networks:",datasetname1,datasetname2))
  analysisdataname<-analysisdataset_names[dataset_index]
  manual1 <- manuals_1[dataset_index]
  manual2 <- manuals_2[dataset_index]
  
  pairwise_metrics_frame<-data.frame()
  for(algo1 in algorithms){
    #for(percent in samp_percentages){
    #  percentstring<-as.integer(percent*100)
      for(directed in c(TRUE)){
        for(fitted in c(FALSE)){
          #TODO: manual
          
          #for(directed in c(TRUE,FALSE)){
          #for(fitted in c(TRUE,FALSE)){
          
          for(testnum in (1:N_TESTS)){
            pairwise_metrics_frame<-rbind(pairwise_metrics_frame,
                                          create_pairmetric_row_helper(
                                            datasetname1,datasetname2,analysisdataname,algo1,'manual',directed,fitted,manual1,manual2,
                                            testnum
                                          ))
            
            #pairwise_metrics_frame<-rbind(pairwise_metrics_frame,
            #                              create_pairmetric_row_helper(
            #                                datasetname1,downsamplenum1,randomsamplenum1,percent,bootstramsamplenum1,algo,directed,fitted,manual,
            #                                datasetname2,downsamplenum2,randomsamplenum2,percent,bootstramsamplenum2,algo,directed,fitted,manual,
            #                                BF_score,shd_score,hamming_score,testnum
            #                              ))
          }
          
        }
      }
      
    #}
  }
  
  write.csv(pairwise_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'manualpairwise_','datasetname1_',datasetname1,'_datasetname2_',datasetname2,'_metrics.csv',sep=''))
}





##########
#
# PART TWO: Automatic methods against each other
#
##########

dataset_subset_names <- c(
  'nsf_subset','sal_subset','eopn_subset',
  'nsf_sal','nsf_eopn','sal_eopn','nsf_sal_eopn'
  #'nsf_sal_subset','nsf_eopn_subset','sal_eopn_subset','nsf_sal_eopn_subset'
)


for (datasetname1 in dataset_subset_names){
  for (datasetname2 in dataset_subset_names){
    
    manual1 <- FALSE
    manual2 <- FALSE
    analysisdataname <- datasetname1
    pairwise_metrics_frame<-data.frame()
    for(algo in algorithms){
      #for(percent in samp_percentages){
      #  percentstring<-as.integer(percent*100)
      for(directed in c(TRUE)){
        for(fitted in c(FALSE)){
          #TODO: manual
          
          #for(directed in c(TRUE,FALSE)){
          #for(fitted in c(TRUE,FALSE)){
          
          for(testnum in (1:N_TESTS)){
            pairwise_metrics_frame<-rbind(pairwise_metrics_frame,
                                          create_pairmetric_row_helper(
                                            datasetname1,datasetname2,analysisdataname,algo,algo,directed,fitted,manual1,manual2,
                                            testnum
                                          ))
            
            #pairwise_metrics_frame<-rbind(pairwise_metrics_frame,
            #                              create_pairmetric_row_helper(
            #                                datasetname1,downsamplenum1,randomsamplenum1,percent,bootstramsamplenum1,algo,directed,fitted,manual,
            #                                datasetname2,downsamplenum2,randomsamplenum2,percent,bootstramsamplenum2,algo,directed,fitted,manual,
            #                                BF_score,shd_score,hamming_score,testnum
            #                              ))
          }
          
        }
      }
      
      #}
    }
    
    write.csv(pairwise_metrics_frame,paste(EVAL_METRICS_DIRECTORY,'fullpairwise_','datasetname1_',datasetname1,'_datasetname2_',datasetname2,'_metrics.csv',sep=''))
  } 
}

  
  

##########
#
# END
#
##########



##########
#
# Part 3: printing edges
#
##########
dataset_names_1 <- c(
  'nsf_subset','sal_subset','eopn_subset',
  'nsf_full','sal_full','eopn_full',
  'nsf_sal','nsf_eopn','sal_eopn','nsf_sal_eopn'
  #'nsf_sal_subset','nsf_eopn_subset','sal_eopn_subset','nsf_sal_eopn_subset'
)
dataset_max_sizes <- c(
  697,594,1274,
  697,594,1274,
  594*2,697*2,594*2,594*3
)
analysisdataset_names <- c('nsf_subset','sal_subset','eopn_subset',
                           'nsf_full','sal_full','eopn_full')

edges_frame<-data.frame()
for(dataset_index in 1:length(dataset_names_1)){
  datasetname1<-dataset_names_1[dataset_index]
  n_points<-dataset_max_sizes[dataset_index]
  print(paste("Dataset name:",datasetname1))
  for(algo1 in algorithms){
    print(paste("Algo:",algo1))
    #for(percent in samp_percentages){
    #  percentstring<-as.integer(percent*100)
    algotemp<-data.frame()
    for(directed in c(TRUE)){
      for(fitted in c(FALSE)){
        for(bootstrap_num in 1:5){
          print(paste("Bootstrap num:",bootstrap_num))
          for(randomsample_num in 1:5){
            print(paste("Random num:",randomsample_num))
            for(downsample_num in 1:N_DOWNSAMPLED_SAMPLES){
              g<-read_graph(datasetname1,downsample_num,randomsample_num,n_points,bootstrap_num,algo1,directed,fitted,FALSE)
              
              garcs<-arcs(g)
              for(r in 1:nrow(garcs)){
                edge<-garcs[r,]
                
                row<-data.frame(
                  datasetname=datasetname1,
                  downsample_number=downsample_num,
                  randomsample_number=randomsample_num,
                  bootstrap_num=bootstrap_num,
                  n=n_points,
                  algorithm=algo1,
                  directed=directed,
                  fitted=fitted,
                  from=edge['from'],
                  to=edge['to']
                )  
                algotemp<-rbind(algotemp,row)
              }
              
              
              
            }
          }
        }
        
      }
    }
    edges_frame<-rbind(edges_frame,algotemp)
    print(nrow(edges_frame))
    
    #}
  }
  
  
}
write.csv(edges_frame,paste(EVAL_METRICS_DIRECTORY,'edges.csv',sep=''))
