
#threshold confidence level to keep
confidence_cut_category = 0.4
confidence_cut_topic = 0.2

#which portion of top scores to take, takes 1/parameter highest scores
score_cut_category = 3
score_cut_topic = 3


import numpy as np
import pandas as p
import pickle
import os


#directory path
path = os.getcwd() + '/'
tables_path = path + "tables/"
dictionaries_path = path + "dicts/"

def create_dicts(confidence_cut_category, confidence_cut_topic, score_cut_category, score_cut_topic):
    
    #load relevant tables
    document_on_ad_document_ctr = p.read_csv(tables_path + 'document_on_ad_document_ctr.csv')
    documents_categories = p.read_csv(tables_path + 'documents_categories.csv')
    documents_topics = p.read_csv(tables_path + 'documents_topics.csv')


    #get only highest confidence topic and category
    max_rows = documents_categories.groupby(['document_id'])['confidence_level'].transform(max) == documents_categories['confidence_level']
    documents_categories = documents_categories[max_rows].drop_duplicates(subset = ['document_id','confidence_level'],keep = 'last')
    max_rows = documents_topics.groupby(['document_id'])['confidence_level'].transform(max) == documents_topics['confidence_level']
    documents_topics = documents_topics[max_rows].drop_duplicates(subset = ['document_id','confidence_level'],keep = 'last')
    del max_rows


    #remove category ids below a certain confidence level and merge promoted with documents_categories_reduced
    documents_categories_reduced = documents_categories[documents_categories['confidence_level'] > confidence_cut_category].drop("confidence_level", axis = 1)

    print ('category ids percent after cutting confidence threshold: ' + repr(float(len(documents_categories_reduced.category_id.unique())) / len(documents_categories.category_id.unique())))


    #take only highest scored ad_on_doc
    document_on_ad_document_ctr_categories = document_on_ad_document_ctr[: int (document_on_ad_document_ctr.shape[0] / score_cut_category)]

    print ('precentage left after category score threshold: ' + repr(float(document_on_ad_document_ctr_categories.shape[0]) / document_on_ad_document_ctr.shape[0]))
    print ('minimal score taken: ' + repr(document_on_ad_document_ctr_categories.score_docXad_doc.min()))


    #merge with category ids of doc and ad
    document_on_ad_document_ctr_categories = document_on_ad_document_ctr_categories.merge(documents_categories_reduced, how='left', on='document_id')
    documents_categories_reduced.rename(columns={'document_id': 'ad_document_id'}, inplace=True)
    document_on_ad_document_ctr_categories = document_on_ad_document_ctr_categories.merge(documents_categories_reduced, how='left', on='ad_document_id', suffixes=('_doc','_ad'))


    document_on_ad_document_ctr_categories.isnull().sum()


    #drop lines with nulls, ad_id and doc_id columns
    document_on_ad_document_ctr_categories = document_on_ad_document_ctr_categories.dropna()
    document_on_ad_document_ctr_categories.drop(document_on_ad_document_ctr_categories.columns[[0,1]],axis = 1,inplace=True)
    document_on_ad_document_ctr_categories.reset_index(drop=True, inplace=True)


    #put lower category_id of both on left side
    for i, row in enumerate(document_on_ad_document_ctr_categories.itertuples()):
        if row.category_id_doc > row.category_id_ad:
            ad = row.category_id_ad
            doc = row.category_id_doc
            document_on_ad_document_ctr_categories.set_value(i,'category_id_doc', ad)
            document_on_ad_document_ctr_categories.set_value(i,'category_id_ad', doc)
    document_on_ad_document_ctr_categories.rename(columns={'category_id_doc' : 'category_id_l', 'category_id_ad' : 'category_id_r'},inplace=True)


    #count how many times each pair shows, and reduce by number of shows
    #keep top 20% score even if count is 1
    high_score = document_on_ad_document_ctr_categories.score_docXad_doc.quantile(q=0.8, interpolation='higher')
    doc_ad_doc_count = document_on_ad_document_ctr_categories.groupby(['category_id_l','category_id_r']).score_docXad_doc.agg({'mean_score' : 'mean', 'count' : 'count'}).reset_index()
    doc_ad_doc_count = doc_ad_doc_count[(doc_ad_doc_count['count'] > 1) | (doc_ad_doc_count['mean_score'] > high_score)]


    #create a dictionary, for each tuple (x_category,y_category) where x_category < y_category return the mean_score
    dict_category = {}
    for row in doc_ad_doc_count.itertuples():
        key = (row.category_id_l,row.category_id_r)
        key_r = (row.category_id_r,row.category_id_l)
        score = row.mean_score
        dict_category[key] = score
        dict_category[key_r] = score


    #save dictionary to file
    category_dict_name = 'dict_category_' + repr(confidence_cut_category) + '_' + repr(score_cut_category)

    with open(dictionaries_path + category_dict_name, 'ab') as handle:
        pickle.dump(dict_category, handle)


    print ("categories correlations dictionary created in " + dictionaries_path + " directory")

    #remove category ids below a certain confidence level and merge promoted with documents_categories_reduced
    documents_topics_reduced = documents_topics[documents_topics['confidence_level'] > confidence_cut_topic].drop("confidence_level", axis = 1)
 
    print ('topics ids percent after cutting confidence threshold: ' + repr(float(len(documents_topics_reduced.topic_id.unique())) / len(documents_topics.topic_id.unique())))


    #take only highest scored ad_on_doc
    document_on_ad_document_ctr_topics = document_on_ad_document_ctr[: int (document_on_ad_document_ctr.shape[0] / score_cut_topic)]
    print ('precentage left after topic score threshold: ' + repr(float(document_on_ad_document_ctr_topics.shape[0]) / document_on_ad_document_ctr.shape[0]))
    print ('minimal score taken: ' + repr(document_on_ad_document_ctr_topics.score_docXad_doc.min()))


    #merge with category ids of doc and ad
    document_on_ad_document_ctr_topics = document_on_ad_document_ctr_topics.merge(documents_topics_reduced, how='left', on='document_id')
    documents_topics_reduced.rename(columns={'document_id': 'ad_document_id'}, inplace=True)
    document_on_ad_document_ctr_topics = document_on_ad_document_ctr_topics.merge(documents_topics_reduced, how='left', on='ad_document_id', suffixes=('_doc','_ad'))


    #drop lines with nulls, ad_id and doc_id columns
    document_on_ad_document_ctr_topics = document_on_ad_document_ctr_topics.dropna()
    document_on_ad_document_ctr_topics.drop(document_on_ad_document_ctr_topics.columns[[0,1]],axis = 1,inplace=True)
    document_on_ad_document_ctr_topics.reset_index(drop=True, inplace=True)


    #put lower category_id of both on left side
    for i, row in enumerate(document_on_ad_document_ctr_topics.itertuples()):
        if row.topic_id_doc > row.topic_id_ad:
            ad = row.topic_id_ad
            doc = row.topic_id_doc
            document_on_ad_document_ctr_topics.set_value(i,'topic_id_doc', ad)
            document_on_ad_document_ctr_topics.set_value(i,'topic_id_ad', doc)
    document_on_ad_document_ctr_topics.rename(columns={'topic_id_doc' : 'topic_id_l', 'topic_id_ad' : 'topic_id_r'},inplace=True)


    #count how many times each pair shows, and reduce by number of shows
    #keep top 20% score even if count is 1

    high_score = document_on_ad_document_ctr_topics.score_docXad_doc.quantile(q=0.8, interpolation='higher')
    doc_ad_doc_count = document_on_ad_document_ctr_topics.groupby(['topic_id_l','topic_id_r']).score_docXad_doc.agg({'mean_score' : 'mean', 'count' : 'count'}).reset_index()
    doc_ad_doc_count = doc_ad_doc_count[(doc_ad_doc_count['count'] > 1) | (doc_ad_doc_count['mean_score'] > high_score)]
    

    #create a dictionary, for each tuple (x_category,y_category) where x_category < y_category return the mean_score
    dict_topic = {}
    for row in doc_ad_doc_count.itertuples():
        key = (row.topic_id_l,row.topic_id_r)
        key_r = (row.topic_id_r,row.topic_id_l)
        score = row.mean_score
        dict_topic[key] = score
        dict_topic[key_r] = score


    #save dictionary to file
    topic_dict_name = 'dict_topic_' + repr(confidence_cut_topic) + '_' + repr(score_cut_topic)

    with open(dictionaries_path + topic_dict_name, 'ab') as handle:
        pickle.dump(dict_topic, handle)

        
    print ("topics correlations dictionary created in " + dictionaries_path + " directory")

    return category_dict_name, topic_dict_name
