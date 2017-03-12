
import os
import pandas as p

#directory path
path = os.getcwd() + "/"
tables_path = path + "tables/"

events = p.read_csv(tables_path + "events.csv", usecols = ["display_id", "document_id", "platform"], dtype={"platform":str})

#cast all platform to int, change missing values to median
events.loc[events['platform'] == '\\N', 'platform'] = events['platform'][events['platform'] != '\\N'].median()
events['platform'] = events['platform'].apply(int)

    
#prepare one-hot for platform
platform_dummies = p.get_dummies(events['platform'],prefix="plat").astype(int)
events = events.drop('platform',axis = 1).join(platform_dummies)
del platform_dummies

#export table
events.to_csv(tables_path + 'platform.csv', index=False)
print ("platform.csv created in " + tables_path + " directory")

