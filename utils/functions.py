import numpy as np
import pandas as p
import os

#directory path
path = os.getcwd() + "/"
tables_path = path + "tables/"


def predict(alg, test, predictors):
    predY = list(alg.predict_proba(test[predictors]).astype(float)[:,1])
    predict = np.asarray(predY)
    test_copy = test.copy()
    test_copy['predict'] = predict
    map_score = score_map(test_copy)
    portion_score = score_portion(test_copy)
    
    return map_score, portion_score

#scoring function calculating portion of correct display ids predictions
def score_portion(val_copy):
    max_rows = val_copy.groupby(['display_id'])['predict'].transform(max) == val_copy['predict']
    final = val_copy[max_rows]
    success = final[final['clicked'] == True]
    score = float(len(success)) / float(len(final))
    print("PORTION: %.12f" % score)
    return score


#scoring function taking in consideration the distance of right ad from 1st position
def score_map(val_copy):
    val_copy.sort_values(['display_id', 'predict'], inplace=True, ascending=[True, False] )
    val_copy["seq"] = np.arange(val_copy.shape[0])
    Y_seq = val_copy[val_copy.clicked == 1].seq.values
    Y_first = val_copy[['display_id', 'seq']].drop_duplicates(subset='display_id', keep='first').seq.values
    Y_ranks = Y_seq - Y_first
    score = np.mean(1.0 / (1.0 + Y_ranks))
    print("MAP: %.12f" % score)
    return score

#get random part of train for fast computing and testing
def fractioned(train, test, fraction):
    display_ids = train.groupby(['display_id'])['display_id'].agg({'count' : 'count'}).reset_index().drop('count',axis = 1)
    chosen_displays = display_ids.sample(frac = fraction)
    train = chosen_displays.merge(train, how = 'inner', on = 'display_id')

    #same for test
    display_ids = test.groupby(['display_id'])['display_id'].agg({'count' : 'count'}).reset_index().drop('count',axis = 1)
    chosen_displays = display_ids.sample(frac = fraction)
    test = chosen_displays.merge(test, how = 'inner', on = 'display_id')
    return train, test

#get a correlation score of the ads document topic on the displays document topic, and same for the category
def correlations(train, test, top_dict, cat_dict):
    dictionary, id_ad, id_doc, confi_ad, confi_doc, corel = top_dict, 'topic_id_ad', 'topic_id_doc', 'confi_top_ad', 'confi_top_doc', 'cor_top'

    for i in range(2):
        #get all pairs of topic/category of ad document and displays document, from train and test
        correlations = train[[id_ad,id_doc]].groupby([id_ad,id_doc]).count().reset_index()
        correlations = correlations.merge(test[[id_ad,id_doc]].groupby([id_ad,id_doc]).count().reset_index(), how = 'outer', on = [id_ad,id_doc])

        #order these pairs in tuples for dictionary use
        correlations['tup'] = list(zip(correlations[id_ad], correlations[id_doc]))

        #get the correlation scores through the dictionary
        correlations[corel] = correlations['tup'].map(dictionary)

        #remove tup column
        correlations.drop('tup',axis = 1,inplace=True)

        #fill NAs with median
        correlations = correlations.fillna(correlations[corel].median())

        #merge these correlations with train and test
        train = train.merge(correlations, how = 'left', on = [id_ad,id_doc])
        test = test.merge(correlations, how = 'left', on = [id_ad,id_doc])

        #multiply the correlation by confidence scores of ad and doc
        train[corel] = train[corel] * train[confi_ad] * train[confi_doc]
        test[corel] = test[corel] * test[confi_ad] * test[confi_doc]

        #do the same now for the categories on next loop
        dictionary, id_ad, id_doc, confi_ad, confi_doc, corel = cat_dict, 'category_id_ad', 'category_id_doc', 'confi_cat_ad', 'confi_cat_doc', 'cor_cat'
    return train, test


#merge ctr & time features into train and test
def merge_ctrs_and_time(train, test):
    
    #load each of the CTR and time tables, merge with train and test and delete the pre-merged feature table
    
    #AD_ID CTR
    ad_ctr = p.read_csv(tables_path + 'ad_ctr.csv')
    train = train.merge(ad_ctr, how = 'left', on = 'ad_id')
    test = test.merge(ad_ctr, how = 'left', on = 'ad_id')
    del ad_ctr
    
    #AD_IDS' - DOCUMENT CTR
    ad_document_ctr = p.read_csv(tables_path + 'ad_document_ctr.csv')
    train = train.merge(ad_document_ctr, how = 'left', on = 'ad_document_id')
    test = test.merge(ad_document_ctr, how = 'left', on = 'ad_document_id')
    del ad_document_ctr
    
    #ADVERTISER_ID CTR
    advertiser_ctr = p.read_csv(tables_path + 'advertiser_ctr.csv')
    train = train.merge(advertiser_ctr, how = 'left', on = 'advertiser_id')
    test = test.merge(advertiser_ctr, how = 'left', on = 'advertiser_id')
    del advertiser_ctr

    #CAMPAIGN_ID CTR
    campaign_ctr = p.read_csv(tables_path + 'campaign_ctr.csv')
    train = train.merge(campaign_ctr, how = 'left', on = 'campaign_id')
    test = test.merge(campaign_ctr, how = 'left', on = 'campaign_id')
    del campaign_ctr
    
    #DOCUMENT_ID of DISPLAY on AD_ID CTR
    document_on_ad_ctr = p.read_csv(tables_path + 'document_on_ad_ctr.csv')
    train = train.merge(document_on_ad_ctr, how = 'left', on = ['document_id', 'ad_id'])
    test = test.merge(document_on_ad_ctr, how = 'left', on = ['document_id', 'ad_id'])
    del document_on_ad_ctr
    
    #DOCUMENT_ID of DISPLAY on AD_IDS' DOCUMENT CTR
    document_on_ad_document_ctr = p.read_csv(tables_path + 'document_on_ad_document_ctr.csv')
    train = train.merge(document_on_ad_document_ctr, how = 'left', on = ['document_id', 'ad_document_id'])
    test = test.merge(document_on_ad_document_ctr, how = 'left', on = ['document_id', 'ad_document_id'])
    del document_on_ad_document_ctr
    
    #DOCUMENT_ID of DISPLAY on ADVERTISER_ID CTR
    document_on_advertiser_ctr = p.read_csv(tables_path + 'document_on_advertiser_ctr.csv')
    train = train.merge(document_on_advertiser_ctr, how = 'left', on = ['document_id', 'advertiser_id'])
    test = test.merge(document_on_advertiser_ctr, how = 'left', on = ['document_id', 'advertiser_id'])
    del document_on_advertiser_ctr
    
    #DOCUMENT_ID of DISPLAY on CAMPAIGN_ID CTR
    document_on_campaign_ctr = p.read_csv(tables_path + 'document_on_campaign_ctr.csv')
    train = train.merge(document_on_campaign_ctr, how = 'left', on = ['document_id', 'campaign_id'])
    test = test.merge(document_on_campaign_ctr, how = 'left', on = ['document_id', 'campaign_id'])
    del document_on_campaign_ctr
    
    #TIME TABLE OF WHEN DID THE DISPLAY OCCUR (TIME OF DAY AND MIDWEEK/WEEKEND)
    time_table = p.read_csv(tables_path + 'time_table.csv')
    train = train.merge(time_table, how = 'left', on = 'display_id')
    test = test.merge(time_table, how = 'left', on = 'display_id')
    del time_table
    
    return train, test


#get percentage of missing values in table
def missing_values_table(df): 
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = p.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns
