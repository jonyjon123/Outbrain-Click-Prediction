
import os
import pandas as p

#directory path
path = os.getcwd() + "/"
tables_path = path + "tables/"

promoted = p.read_csv(tables_path + "promoted_content.csv")
topics_categories = p.read_csv(tables_path + "topics_categories.csv")

#create promoted, promoted content table (ad information) with its' categories and topics
promoted = promoted.merge(topics_categories, how='left', on='document_id')
promoted.rename(columns={'document_id': 'ad_document_id'}, inplace=True)
    
#assign nulls to -1 and cast the ids back to int (the merge changes them to float)
promoted[['confi_cat', 'confi_top']] = promoted[['confi_cat', 'confi_top']].fillna(0)
promoted = promoted.fillna(-1)
promoted[['topic_id', 'category_id']] = promoted[['topic_id', 'category_id']].astype(int)

#export table
promoted.to_csv(tables_path + 'promoted_content_prep.csv', index=False)
print ("promoted_content_prep.csv created in " + tables_path + " directory")

