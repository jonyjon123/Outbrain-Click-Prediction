import os
import pandas as p

#directory path
path = os.getcwd() + "/"
tables_path = path + "tables/"

#read source tables
documents_topics = p.read_csv(tables_path + "documents_topics.csv")
documents_categories = p.read_csv(tables_path + "documents_categories.csv")


#rename columns of repeating names
documents_topics.rename(columns={'confidence_level': 'confi_top'}, inplace=True)
documents_categories.rename(columns={'confidence_level': 'confi_cat'}, inplace=True)



#for each document- leave only topic/category with the highest confidence level
max_rows = documents_categories.groupby(['document_id'])['confi_cat'].transform(max) == documents_categories['confi_cat']
documents_categories = documents_categories[max_rows].drop_duplicates(subset = ['document_id','confi_cat'],keep = 'last')
max_rows = documents_topics.groupby(['document_id'])['confi_top'].transform(max) == documents_topics['confi_top']
documents_topics = documents_topics[max_rows].drop_duplicates(subset = ['document_id','confi_top'],keep = 'last')



#create one table for both categories and topics
topics_categories = documents_topics.merge(documents_categories, how = 'outer', on = 'document_id')
del max_rows, documents_categories, documents_topics


#assign nulls to -1 and cast the ids back to int (the merge changes them to float)
topics_categories[['confi_cat', 'confi_top']] = topics_categories[['confi_cat', 'confi_top']].fillna(0)
topics_categories = topics_categories.fillna(-1)
topics_categories[['document_id', 'topic_id', 'category_id']] = topics_categories[['document_id', 'topic_id', 'category_id']].astype(int)


#export table
topics_categories.to_csv(tables_path + 'topics_categories.csv', index=False)
print ("topics_categories.csv created in " + tables_path + " directory")

