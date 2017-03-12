import numpy as np
import pandas as p
import os

path = os.getcwd() + "/"
tables_path = path + "tables/"

#contains the display_ids (page shown to a user) with all the ads on the page and whether they were clicked or not (always one was clicked)
train_ctr = p.read_csv(tables_path + "train.csv")

#contains the document_id of the display (the webpage visited)
events = p.read_csv(tables_path + "events.csv", usecols = [0,1])

#contains information about the ads, document_id - the webpage the ad is leading to, campaign_id - the campaign of the ad, advertiser_id - the advertiser of the ad
promoted_content = p.read_csv(tables_path + "promoted_content.csv")
promoted_content.rename(columns={'document_id': 'ad_document_id'}, inplace=True)


#function for calculating the clicks through rate (clicked percentage) over different features
def ctr(df, over, name):

    #'over' are the features we get CTR on (can be a single feature or a couple)

    #name of the feature
    name = "score_" + name

    #columns to be dropped later on
    to_drop1 = ['ads_on_doc','clicked','uni_chance','clicked_percent','clicked_percent_normalized','likelihood_normalized']
    to_drop2 = ['total','like_mul_total_normalized']

    #expected chance of ad to be chosen [uni_chance]
    #ads_on_doc is a number resembling how many ads were on the display
    df['uni_chance'] = 1 / df['ads_on_doc']

    #actual clicked precentage
    df['clicked_percent'] = df['clicked'] / df['total']

    #normalized clicked precentage
    df['clicked_percent_normalized'] = (df['clicked'] + 12 * df['clicked_percent'].mean()) / (12 + df['total'])

    #create a likelihood column which shows how strong the normalized 
    #clicked percentage is in relation to the uniform chance
    df['likelihood_normalized'] = df['clicked_percent_normalized'] / df['uni_chance']

    #in order to make a mean from different ad_on_doc but same 'over' columns
    df['like_mul_total_normalized'] = df['likelihood_normalized'] * df['total']

    #get scores for 'over' grouped across the different ad_on_doc
    df = df.drop(to_drop1,axis = 1).groupby(over).sum().reset_index()
    df[name] = df['like_mul_total_normalized'] / df['total']

    df.drop(to_drop2,axis = 1, inplace=True)
    df.sort_values(name,inplace=True, ascending=False)
    return df


#add number of ads on display ['ads_on_doc']
train_ad_count_per_display = train_ctr.groupby(['display_id'])['display_id'].agg({'ads_on_doc' : 'count'}).reset_index()
train_ctr = train_ctr.merge(train_ad_count_per_display, how = 'left', on = 'display_id')
del train_ad_count_per_display


#add the document_id of the display
train_ctr = train_ctr.merge(events, how = 'left', on = 'display_id')

#add the document_id describing ad
train_ctr = train_ctr.merge(promoted_content, how = 'left', on = 'ad_id')


#normalized CTR of ad
ad_ctr = train_ctr.groupby(['ad_id','ads_on_doc']).clicked.agg({'clicked' : 'sum', 'total' : 'count'}).reset_index()

#normalized CTR of ads' document
ad_document_ctr = train_ctr.groupby(['ad_document_id','ads_on_doc']).clicked.agg({'clicked' : 'sum', 'total' : 'count'}).reset_index()

#normalized CTR of document coupled with ad 
document_on_ad_ctr = train_ctr.groupby(['document_id','ad_id','ads_on_doc']).clicked.agg({'clicked' : 'sum', 'total' : 'count'}).reset_index()

#normalized CTR of displays' document coupled with ads' document
document_on_ad_document_ctr = train_ctr.groupby(['document_id','ad_document_id','ads_on_doc']).clicked.agg({'clicked' : 'sum', 'total' : 'count'}).reset_index()

#normalized CTR of advertiser alone
advertiser_ctr = train_ctr.groupby(['advertiser_id','ads_on_doc']).clicked.agg({'clicked' : 'sum', 'total' : 'count'}).reset_index()

#normalized CTR of campaign alone
campaign_ctr = train_ctr.groupby(['campaign_id','ads_on_doc']).clicked.agg({'clicked' : 'sum', 'total' : 'count'}).reset_index()

#normalized CTR of advertiser coupled with displays' document
document_on_advertiser_ctr = train_ctr.groupby(['advertiser_id','document_id','ads_on_doc']).clicked.agg({'clicked' : 'sum', 'total' : 'count'}).reset_index()

#normalized CTR of campaign coupled with displays' document
document_on_campaign_ctr = train_ctr.groupby(['campaign_id','document_id','ads_on_doc']).clicked.agg({'clicked' : 'sum', 'total' : 'count'}).reset_index()


#create all of the final tables using ctr function
ad_ctr = ctr(ad_ctr, ['ad_id'], 'ad')
ad_ctr.to_csv(tables_path + 'ad_ctr.csv', index = False)

ad_document_ctr = ctr(ad_document_ctr, ['ad_document_id'], 'ad_doc')
ad_document_ctr.to_csv(tables_path + 'ad_document_ctr.csv', index = False)

document_on_ad_ctr = ctr(document_on_ad_ctr, ['document_id', 'ad_id'], 'docXad')
document_on_ad_ctr.to_csv(tables_path + 'document_on_ad_ctr.csv', index=False)

document_on_ad_document_ctr = ctr(document_on_ad_document_ctr, ['document_id', 'ad_document_id'], 'docXad_doc')
document_on_ad_document_ctr.to_csv(tables_path + 'document_on_ad_document_ctr.csv', index=False)

advertiser_ctr = ctr(advertiser_ctr, ['advertiser_id'], 'adv')
advertiser_ctr.to_csv(tables_path + 'advertiser_ctr.csv', index = False)

campaign_ctr = ctr(campaign_ctr, ['campaign_id'], 'camp')
campaign_ctr.to_csv(tables_path + 'campaign_ctr.csv', index = False)

document_on_advertiser_ctr = ctr(document_on_advertiser_ctr, ['document_id', 'advertiser_id'], 'docXadv')
document_on_advertiser_ctr.to_csv(tables_path + 'document_on_advertiser_ctr.csv', index = False)

document_on_campaign_ctr = ctr(document_on_campaign_ctr, ['document_id', 'campaign_id'], 'docXcamp')
document_on_campaign_ctr.to_csv(tables_path + 'document_on_campaign_ctr.csv', index = False)


print ("ctr tables created in " + tables_path + " directory")