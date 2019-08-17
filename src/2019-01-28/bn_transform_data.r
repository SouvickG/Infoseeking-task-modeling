#TODO
#1) Get full and slice data of: NSF, SAL, EOPN
#2) Normalize columns: transform and rename for: NSF, SAL, EOPN
#2) Make sure data is correct (correct queries/pages) for: SAL, EOPN
#3) Transform data types


N_DOWNSAMPLED_SAMPLES <- 20
N_RANDOM_SAMPLES <- 20
N_RANDOM_BASE <- 100
N_RANDOM_INCREMENTS <- 100
DATA_DIRECTORY <- '/Users/Matt/Desktop/Experiment/data/interim/graph_data_samples/'
nsf_fulldata_dir<-'/Users/Matt/Desktop/Experiment/data/raw/features_nsf_byquery.csv'
sal_fulldata_dir<-'/Users/Matt/Desktop/Experiment/data/raw/features_sal_byquery_cleaned.csv'
eopn_fulldata_dir<-'/Users/Matt/Desktop/Experiment/data/raw/features_eopn_byquery_cleaned.csv'

#samp_percs <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)

#TODO: Proper transformation
#TODO: Transfer to all data sets (but don't transform variables multiple times!)
transform_data <- function(data,framename){
  data[] <- lapply(data, function(x) {
    if(is.factor(x) || is.integer(x)) as.numeric(as.character(x)) else x
  })
  
  data$pages_num_segment<-scale(data$pages_num_segment)
  data$time_dwelltotal_total_content_segment<-scale(data$time_dwelltotal_total_content_segment)
  data$time_dwelltotal_total_serp_segment<-scale(data$time_dwelltotal_total_serp_segment)
  data$query_length<-scale(data$query_length)
  
  return(data);
  
  #Features to normalize:
  #Same scale: search_expertise, search_frequency, search_journalism, search_years, topic_familiarity, assignment_experience, post_difficult, post_rushed
  #unit mean, std: all the above, 'pages_num_segment','time_dwelltotal_total_content_segment','time_dwelltotal_total_serp_segment','query_length'
}


rename_columns <- function(data){
  colnames(data)[colnames(data)=="post_difficult"] <- "task_difficulty"
  
  #'facet_goal_val_amorphous',
  #'topic_globalwarming',
  #'facet_product_val_intellectual',
  #'search_expertise',
  #'search_frequency',
  #'search_journalism',
  #'search_years',
  #'topic_familiarity',
  #'assignment_experience',
  #'post_difficult',
  #'post_rushed',
  #'pages_num_segment'
  #'time_dwelltotal_total_content_segment',
  #'time_dwelltotal_total_serp_segment',
  #'query_length'
  #'
  #'
  return(data);
}


nsf_exclusive_cols<-c(
  'facet_goal_val_amorphous',
  'facet_product_val_intellectual',
  'assignment_experience',
  'post_rushed',
  'search_expertise',
  'search_frequency',
  'search_journalism',
  'search_years',
  'task_difficulty',
  'topic_familiarity',
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
sal_exclusive_cols <- c(
  'facet_goal_val_amorphous',
  'facet_product_val_intellectual',
  'search_frequency',
  'search_years',
  'topic_familiarity',
  'search_expertise',
  'task_difficulty',
  
  'pages_num_segment',
  'time_dwelltotal_total_content_segment',
  'time_dwelltotal_total_serp_segment',
  'query_length'
)

eopn_exclusive_cols <- c(
  
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



nsf_missing_cols<-c(
)
sal_missing_cols <- c(
  'assignment_experience',
  'search_journalism',
  'post_rushed'
)

eopn_missing_cols <- c(
  'assignment_experience',
  'search_frequency',
  'search_journalism',
  'search_years'
)


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
  'topic_familiarity',
  'search_years',
  'search_frequency',
  'search_expertise',
  'task_difficulty',
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


# Get full data and subset for each data set
# DON'T TRANSFORM YET!  NEEDS TO BE DONE RELATIVELY!
nsf_data_full <- read.csv(nsf_fulldata_dir)
nsf_data_full <- rename_columns(nsf_data_full)
nsf_data_subset <- transform_data(nsf_data_full[,shared_cols])
nsf_data_allcolumns <- data.frame(nsf_data_full)
nsf_data_allcolumns[,nsf_missing_cols] <- NA


sal_data_full<-read.csv(sal_fulldata_dir)
sal_data_full <- rename_columns(sal_data_full)
sal_data_subset <- transform_data(sal_data_full[,shared_cols])
sal_data_allcolumns <- data.frame(sal_data_full)
sal_data_allcolumns[,sal_missing_cols] <- NA

eopn_data_full<-read.csv(eopn_fulldata_dir)
eopn_data_full <- rename_columns(eopn_data_full)
eopn_data_subset <- transform_data(eopn_data_full[,shared_cols])
eopn_data_allcolumns <- data.frame(eopn_data_full)
eopn_data_allcolumns[,eopn_missing_cols] <- NA


nsf_data_allcolumns<-nsf_data_allcolumns[nsf_exclusive_cols]
sal_data_allcolumns<-sal_data_allcolumns[nsf_exclusive_cols]
eopn_data_allcolumns<-eopn_data_allcolumns[nsf_exclusive_cols]
write.csv(nsf_data_allcolumns,'/Users/Matt/Desktop/nsf_datacols.csv')
write.csv(sal_data_allcolumns,'/Users/Matt/Desktop/sal_datacols.csv')
write.csv(eopn_data_allcolumns,'/Users/Matt/Desktop/eopn_datacols.csv')


transform_data(nsf_data_allcolumns)

nsf_data_exclusive <- nsf_data_allcolumns[nsf_exclusive_cols]
nsf_data_exclusive <- transform_data(nsf_data_exclusive)
write.csv(nsf_data_exclusive,'/Users/Matt/Desktop/nsf_exclusive_scaled.csv')

sal_data_exclusive <- sal_data_allcolumns[sal_exclusive_cols]
sal_data_exclusive <- transform_data(sal_data_exclusive)
write.csv(sal_data_exclusive,'/Users/Matt/Desktop/sal_exclusive_scaled.csv')

eopn_data_exclusive <- eopn_data_allcolumns[eopn_exclusive_cols]
eopn_data_exclusive <- transform_data(eopn_data_exclusive)
write.csv(eopn_data_exclusive,'/Users/Matt/Desktop/eopn_exclusive_scaled.csv')






nsf_sal_data_allcolumns <-rbind(nsf_data_allcolumns,sal_data_allcolumns)
nsf_sal_data_allcolumns <- transform_data(nsf_sal_data_allcolumns)
write.csv(nsf_sal_data_allcolumns,'/Users/Matt/Desktop/nsf_sal_allcolumns_scaled.csv')

nsf_eopn_data_allcolumns <-rbind(nsf_data_allcolumns,eopn_data_allcolumns)
nsf_eopn_data_allcolumns <- transform_data(nsf_eopn_data_allcolumns)
write.csv(nsf_eopn_data_allcolumns,'/Users/Matt/Desktop/nsf_eopn_allcolumns_scaled.csv')

nsf_sal_eopn_data_allcolumns <-rbind(nsf_data_allcolumns,sal_data_allcolumns,eopn_data_allcolumns)
nsf_sal_eopn_data_allcolumns <- transform_data(nsf_sal_eopn_data_allcolumns)
write.csv(nsf_sal_eopn_data_allcolumns,'/Users/Matt/Desktop/nsf_sal_eopn_allcolumns_scaled.csv')

sal_eopn_data_allcolumns <-rbind(sal_data_allcolumns,eopn_data_allcolumns)
sal_eopn_data_allcolumns <- transform_data(sal_eopn_data_allcolumns)
write.csv(sal_eopn_data_allcolumns,'/Users/Matt/Desktop/sal_eopn_allcolumns_scaled.csv')




#
#
#
#
#
# END
#
#
#
#
#
#
merged_data_train <- rbind(nsf_data_allcolumns[nsf_exclusive_cols],eopn_data_allcolumns[nsf_exclusive_cols])
nr<-nrow(sal_data_allcolumns)
np<-200
train_rows <-sample(nr, np)
smp <- sal_data_allcolumns[train_rows, ]
smp_comp <- sal_data_allcolumns[-train_rows, ]

merged_data_train <- rbind(merged_data_train,smp[nsf_exclusive_cols])
merged_data_test <- smp_comp[nsf_exclusive_cols]
write.csv(merged_data_test[nsf_exclusive_cols],'/Users/Matt/Desktop/merged_data_test.csv')


get_dataset_path <-function(dataset_name,downsample_n,n_points,randomsample_n){
  return(paste(DATA_DIRECTORY,dataset_name,'_downsample',downsample_n,'_n',n_points,'_sample',randomsample_n,'.csv',sep=''))
}
# Downsamples
# 1) Downsample + merge
# 2) Transform
# 3) Random sample
# 4) Output
for(downsample_n in (1:N_DOWNSAMPLED_SAMPLES)){
  print(paste("Downsampled sample:",downsample_n))
  # 1) Downsample + Merge
  # 2) TODO: Transform
  # NSF, SAL
  nsf_data_downsample <- nsf_data_subset[sample(nrow(nsf_data_subset), nrow(sal_data_subset)),]
  nsf_sal_merged_data <- rbind(nsf_data_downsample,sal_data_subset)
  
  
  # NSF, EOPN
  eopn_data_downsample <- eopn_data_subset[sample(nrow(eopn_data_subset), nrow(nsf_data_subset)), ]
  nsf_eopn_merged_data <- rbind(eopn_data_downsample,nsf_data_subset)
  
  
  # SAL, EOPN
  eopn_data_downsample <- eopn_data_subset[sample(nrow(eopn_data_subset), nrow(sal_data_subset)), ]
  sal_eopn_merged_data <- rbind(eopn_data_downsample,sal_data_subset)
  
  
  # NSF, SAL, EOPN
  eopn_data_downsample <- eopn_data_subset[sample(nrow(eopn_data_subset), nrow(sal_data_subset)), ]
  nsf_data_downsample <- nsf_data_subset[sample(nrow(nsf_data_subset), nrow(sal_data_subset)),]
  nsf_sal_eopn_merged_data <- rbind(eopn_data_downsample,nsf_data_downsample,sal_data_subset)
  
  
  for(randomsample_n in (1:N_RANDOM_SAMPLES)){
    #print(paste("Random sample:",randomsample_n))

    dataset_names <- c('nsf_sal','nsf_eopn','sal_eopn','nsf_sal_eopn')
    data_supersamples <- list(nsf_sal_merged_data,nsf_eopn_merged_data,sal_eopn_merged_data,nsf_sal_eopn_merged_data)
    
    for(i in 1:length(dataset_names)){
      data_supersample <- data_supersamples[[i]]
      n_total_rows <- nrow(data_supersample)
      dataset_name <- dataset_names[i]
      #print(paste(n_total_rows,dataset_name))
      #print(append(seq(N_RANDOM_BASE,n_total_rows,N_RANDOM_INCREMENTS),n_total_rows))
      for(n_points in append(seq(N_RANDOM_BASE,n_total_rows,N_RANDOM_INCREMENTS),n_total_rows)){
        data_sample <- data_supersample[sample(n_total_rows, n_points), ]
        write.csv(data_sample,get_dataset_path(dataset_name,downsample_n,n_points,randomsample_n))
      }
    }
    
  }
}



nsf_data_full <- transform_data(nsf_data_full[,nsf_fulldata_cols])
sal_data_full <- transform_data(sal_data_full[,sal_fulldata_cols])
eopn_data_full <- transform_data(eopn_data_full[,eopn_fulldata_cols])

print("Original Datasets")
for(randomsample_n in (1:N_RANDOM_SAMPLES)){
  print(paste("Random sample:",randomsample_n))
  dataset_names <- c('eopn_subset','nsf_subset','sal_subset','eopn_full','nsf_full','sal_full')
  data_supersamples <- list(eopn_data_subset,nsf_data_subset,sal_data_subset,eopn_data_full,nsf_data_full,sal_data_full)
  
  for(i in 1:length(dataset_names)){
    data_supersample <- data_supersamples[[i]]
    n_total_rows <- nrow(data_supersample)
    dataset_name <- dataset_names[i]
    #print(paste(n_total_rows,dataset_name))
    for(n_points in append(seq(N_RANDOM_BASE,n_total_rows,N_RANDOM_INCREMENTS),n_total_rows)){
      data_sample <- data_supersample[sample(n_total_rows, n_points), ]
      for(downsample_n in 1:N_DOWNSAMPLED_SAMPLES){
        write.csv(data_sample,get_dataset_path(dataset_name,downsample_n,n_points,randomsample_n))  
      }
    }
  }
  
}
