import numpy as np
import pandas as p
import os

#directory path
path = os.getcwd() + "/"
tables_path = path + "tables/"


#function used to to score prediction in the same way the contest  
def predict(alg, test, predictors):
    predY = list(alg.predict_proba(test[predictors]).astype(float)[:,1])
    predict = np.asarray(predY)
    test_copy = test.copy()
    test_copy['predict'] = predict
    map_score = score_map(test_copy)
    portion_score = score_portion(test_copy)
    
    return map_score, portion_score

def predict_printless(alg, test, predictors):
    predY = list(alg.predict_proba(test[predictors]).astype(float)[:,1])
    predict = np.asarray(predY)
    test_copy = test.copy()
    test_copy['predict'] = predict
    map_score = score_map_printless(test_copy)
    return map_score

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

def score_map_printless(val_copy):
    val_copy.sort_values(['display_id', 'predict'], inplace=True, ascending=[True, False] )
    val_copy["seq"] = np.arange(val_copy.shape[0])
    Y_seq = val_copy[val_copy.clicked == 1].seq.values
    Y_first = val_copy[['display_id', 'seq']].drop_duplicates(subset='display_id', keep='first').seq.values
    Y_ranks = Y_seq - Y_first
    score = np.mean(1.0 / (1.0 + Y_ranks))
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
def missing_values_table(df,topics_categories_flag):  #added flag
        mis_val = df.isnull().sum()
        #added -1s you can remove the comments ron! just for u to see
        if topics_categories_flag:
            topic_id_null = df[df['topic_id_doc'] == -1].shape[0]
            category_id_null = df[df['category_id_doc'] == -1].shape[0]
            mis_val[1] += topic_id_null
            mis_val[2] += topic_id_null
            mis_val[3] += category_id_null
            mis_val[4] += category_id_null
        mis_val_percent = 100 * df.isnull().sum()/len(df)
        mis_val_table = p.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        return mis_val_table_ren_columns
    
#impute missing values in final features table with mean or median values
def fill_na(df,kind):
    if kind == "mean":
        df.score_ad = df.score_ad.fillna(df.score_ad.mean())
        df.score_ad_doc = df.score_ad_doc.fillna(df.score_ad_doc.mean())
        df.score_adv = df.score_adv.fillna(df.score_adv.mean())
        df.score_camp = df.score_camp.fillna(df.score_camp.mean())
        df.score_docXad = df.score_docXad.fillna(df.score_docXad.mean())
        df.score_docXad_doc = df.score_docXad_doc.fillna(df.score_docXad_doc.mean())
        df.score_docXadv = df.score_docXadv.fillna(df.score_docXadv.mean())
        df.score_docXcamp = df.score_docXcamp.fillna(df.score_docXcamp.mean())
    else:
        df.score_ad = df.score_ad.fillna(df.score_ad.median())
        df.score_ad_doc = df.score_ad_doc.fillna(df.score_ad_doc.median())
        df.score_adv = df.score_adv.fillna(df.score_adv.median())
        df.score_camp = df.score_camp.fillna(df.score_camp.median())
        df.score_docXad = df.score_docXad.fillna(df.score_docXad.median())
        df.score_docXad_doc = df.score_docXad_doc.fillna(df.score_docXad_doc.median())
        df.score_docXadv = df.score_docXadv.fillna(df.score_docXadv.median())
        df.score_docXcamp = df.score_docXcamp.fillna(df.score_docXcamp.median())
        
        

#runs grid sreach for given model (random forest, gradient boost, logistic regression) with 2 chosen parameters for each model
def grid_search(alg_name, listx, listy, train, validation, predictors):
    lenx = len(listx)
    leny = len(listy)
    results = [0] * (lenx * leny) 
    if alg_name == "gradient":
        from sklearn.ensemble import GradientBoostingClassifier
        for i in range(leny):
            for j in range(lenx):
                alg = GradientBoostingClassifier(learning_rate=listx[j], max_depth=listy[i]).fit(train[predictors], train["clicked"])
                results[i * leny + j] = predict_printless(alg, validation, predictors)
                print("parameters: learning_rate = " + repr(listx[j]) + ", max_depth = " + repr(listy[i]) + ", score: " + 
                      repr(results[i * leny + j]))
                maxim = np.argmax(results)
        y_i = maxim // lenx
        x_i = maxim % lenx
        best_alg = GradientBoostingClassifier(learning_rate=listx[x_i], max_depth=listy[y_i])
        print("best parameters: learning_rate = " + repr(listx[x_i]) + ", max_depth = " + repr(listy[y_i]) + ", score: " + repr(max(results)))
                      
    
    if alg_name == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        for i in range(leny):
            for j in range(lenx):
                alg = RandomForestClassifier(min_samples_split=listx[j], n_estimators=listy[i]).fit(train[predictors], train["clicked"])
                results[i * leny + j] = predict_printless(alg, validation, predictors)
                print("parameters: min_samples_split = " + repr(listx[j]) + ", n_estimators = " + repr(listy[i]) + ", score: " + repr(results[i * leny + j]))
                maxim = np.argmax(results)
        y_i = maxim // lenx
        x_i = maxim % lenx
        best_alg = RandomForestClassifier(min_samples_split=listx[x_i], n_estimators=listy[y_i])
        print("best parameters: min_samples_split = " + repr(listx[x_i]) + ", n_estimators = " + repr(listy[y_i]) + ", score: " + repr(max(results)))
                      
    
    if alg_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        for i in range(leny):
            for j in range(lenx):
                alg = LogisticRegression(C = listx[j], solver = listy[i]).fit(train[predictors], train["clicked"])
                results[i * leny + j] = predict_printless(alg, validation, predictors)
                print("parameters: C = " + repr(listx[j]) + ", solver = " + repr(listy[i]) + ", score: " + repr(results[i * leny + j]))
                maxim = np.argmax(results)
        y_i = maxim // lenx
        x_i = maxim % lenx
        best_alg = LogisticRegression(C=listx[x_i], solver=listy[y_i])
        print("best parameters: C = " + repr(listx[x_i]) + ", solver = " + repr(listy[y_i]) + ", score: " + repr(max(results)))
        
    return best_alg, max(results)



def feature_selection(train, validation):

    from sklearn.linear_model import LogisticRegression
    alg = LogisticRegression(C = 1e-05, solver = 'sag')
    
    #get best combination of CTRs
    predictors=[x for x in train.columns if x not in ['plat_1', 'plat_2', 'plat_3','cor_cat','cor_top','weekend', 'morning', 'noon', 'evening', 'night','display_id','ad_id','clicked','document_id','platform','ad_document_id','campaign_id','advertiser_id','confi_top_ad','topic_id_ad','topic_id_doc','category_id_ad','confi_cat_ad','confi_top_doc','category_id_doc','confi_cat_doc']]

    i_l = ['','score_ad']
    j_l = ['','score_ad_doc']
    k_l = ['','score_adv']
    l_l = ['','score_camp']
    t_l = ['','score_docXad']
    n_l = ['','score_docXad_doc']
    m_l = ['','score_docXadv']
    p_l = ['','score_docXcamp']
    results = [0] * 256
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for t in range(2):
                        for n in range(2):
                            for m in range(2):
                                for p in range(2):
                                    to_reduce = [i_l[i]] + [j_l[j]] + [k_l[k]] + [l_l[l]] + [t_l[t]] + [n_l[n]] + [m_l[m]] + [p_l[p]]
                                    pred = [x for x in predictors if x not in to_reduce]                               
                                    alg.fit(train[pred], train["clicked"])
                                    results[i + j * 2 + k * 4 + l * 8 + t * 16 + n * 32 + m * 64 + p * 128] = predict_printless(alg, validation, pred)
                                    

    predictors=[x for x in train.columns if x not in ['display_id','ad_id','clicked','document_id','platform','ad_document_id','campaign_id','advertiser_id','confi_top_ad','topic_id_ad','topic_id_doc','category_id_ad','confi_cat_ad','confi_top_doc','category_id_doc','confi_cat_doc']]
    to_reduce = []
    num = np.argmax(results)
    for i in range(8):
        if i == 0:
            if num % 2 == 0:
                to_reduce += [i_l[1]]
        if i == 1:
            if num % 2 == 0:
                to_reduce += [j_l[1]]
        if i == 2:
            if num % 2 == 0:
                to_reduce += [k_l[1]]
        if i == 3:
            if num % 2 == 0:
                to_reduce += [l_l[1]]
        if i == 4:
            if num % 2 == 0:
                to_reduce += [t_l[1]]
        if i == 5:
            if num % 2 == 0:
                to_reduce += [n_l[1]]
        if i == 6:
            if num % 2 == 0:
                to_reduce += [m_l[1]]
        if i == 7:
            if num % 2 == 0:
                to_reduce += [p_l[1]]
        num = num / 2
    #reduce the predictors that don't lead to best score from the CTRs
    predictors = [x for x in predictors if x not in to_reduce]

    #check the rest of the features for best combination
    i_l = [[''],['weekend', 'morning', 'noon', 'evening', 'night']]
    j_l = ['','cor_top']
    k_l = ['','cor_cat']
    l_l = [[''],['plat_1','plat_2','plat_3']]
    t_l = ['','same_topic']
    n_l = ['','same_category']
    results = [0] * 64
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for t in range(2):
                        for n in range(2):
                            to_reduce = [i_l[i]] + [j_l[j]] + [k_l[k]] + l_l[l] + [t_l[t]] + [n_l[n]]
                            pred = [x for x in predictors if x not in to_reduce]
                            alg.fit(train[pred], train["clicked"])
                            results[i + j * 2 + k * 4 + l * 8 + t * 16 + n * 32] = predict_printless(alg, validation, pred)

    num = np.argmax(results)
    to_reduce = []
    for i in range(6):
        if i == 0:
            if num % 2 == 0:
                to_reduce += i_l[1]
        if i == 1:
            if num % 2 == 0:
                to_reduce += [j_l[1]]
        if i == 2:
            if num % 2 == 0:
                to_reduce += [k_l[1]]
        if i == 3:
            if num % 2 == 0:
                to_reduce += l_l[1]
        if i == 4:
            if num % 2 == 0:
                to_reduce += [t_l[1]]
        if i == 5:
            if num % 2 == 0:
                to_reduce += [n_l[1]]
        num = num / 2
    predictors = [x for x in predictors if x not in to_reduce]
    print("best score is: " + repr(max(results)))
    return predictors


def train_validation_split(clicks_train):
    display_ids = clicks_train.groupby(['display_id'])['display_id'].agg({'count' : 'count'}).reset_index().drop('count',axis = 1)
    split = int (display_ids.shape[0] * 4 / 5)
    train = clicks_train.merge(display_ids[:split], how = 'inner', on = 'display_id')
    validation = clicks_train.merge(display_ids[split:], how = 'inner', on = 'display_id')
    return train, validation

def shared_subjects(df):
    df['same_topic'] = df.apply (lambda row: row.topic_id_ad == row.topic_id_doc != -1,axis=1)
    df['same_category'] = df.apply (lambda row: row.category_id_ad == row.category_id_doc != -1,axis=1)
