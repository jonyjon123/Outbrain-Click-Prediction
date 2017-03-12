
#set to 1 if to import shapely and use it, without, some time zone retrievals can not be made
#the rest of the packages are required for time zone fetching
shapely_f = 1

#[pygecoders, geopy, pycountry, pytz, tzwhere, shapley] are the uncommon packages used
import os
import pandas as p
import time
import numpy as np
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from pygeocoder import Geocoder
import pycountry
from pytz import timezone
import pytz
from tzwhere import tzwhere
if shapely_f:
    import shapely


path = os.getcwd() + "/"
tables_path = path + "tables/"


events = p.read_csv(tables_path + "events.csv", usecols = [0,2,4], dtype={"display_id": int, "timestamp" : int, "geo_location" : str})


#functions for parsing the abbreviations from the geo_location
def parse_country(x):
    if type(x) == float: # for nans
        return None
    x = x[:2]
    if x.isdigit():
        return None
    if x == '--':
        return None
    else:
        return x

def parse_state(x):
    if type(x) == float: # for nans
        return None
    x = x[:2] + '-' + x[3:5]
    if len(x) < 4:
        return None
    if x.isdigit():
        return None
    if x == '--':
        return None
    if x == '':
        return None
    else:
        return x

def nan_check(x):
    if x == -1:
        return True
    if type(x) == None:
        return True
    if x != x:
        return True
    return False


#remove rows where geo_location is null, keep events where missing locations will be offset of 0
events_parsing = events[events.geo_location.notnull()]


#dictionaries to get full country and state names
country_mapping = {country.alpha_2 : country.name for country in pycountry.countries}
state_mapping = {country_state.code : country_state.name for country_state in pycountry.subdivisions}


#were missing so added manualy
country_mapping['AN'] = 'Netherlands'
country_mapping['FX'] = 'France'


#get a table of all country - state options ready to pull timezones
#-1 resembles null (replacing for merging purposes)
events_parsing = events_parsing.assign(country = events_parsing['geo_location'].apply(parse_country))
events_parsing = events_parsing.assign(state = events_parsing['geo_location'].apply(parse_state))
events_parsing = events_parsing.assign(country = events_parsing['country'].map(country_mapping))
events_parsing = events_parsing.assign(state = events_parsing['state'].map(state_mapping))
events_parsing = events_parsing[events_parsing.country.notnull()]
events_parsing = events_parsing.fillna(-1)


#table of country and state pairs
country_state_combos = events_parsing[['country','state']].groupby(['country','state']).count().reset_index()


#pull timezones for each country - state pair
#theres' a sleep call every time host cancels connection
gn = Nominatim()

if shapely_f:
    tz = tzwhere.tzwhere(shapely=True, forceTZ=True)
else:
    tz = tzwhere.tzwhere()

total_len = country_state_combos.shape[0]

for i, row in enumerate(country_state_combos.itertuples()):
    nulled = 0
    if nan_check(row.state):
        state = ''
    else:
        state = row.state

    #get the location details
    set = 0
    while(set == 0):
        try:
            location = gn.geocode(row.country + ' ' + state)
            set = 1
            if location == None:
                location = gn.geocode(row.country)
                if location == None:
                    nulled = 1
                    continue
        except:
            time.sleep(5)
    if nulled == 1:
        continue
    #get the time zone details of the location
    if shapely_f:
        timezone_str = tz.tzNameAt(location.latitude, location.longitude, forceTZ=True)
    else:
        timezone_str = tz.tzNameAt(location.latitude, location.longitude)
    if timezone_str == None:
        location = gn.geocode(row.country)
        if shapely_f:
            timezone_str = tz.tzNameAt(location.latitude, location.longitude, forceTZ=True)
        else:
            timezone_str = tz.tzNameAt(location.latitude, location.longitude)
        if timezone_str == None:
            continue
    if timezone_str == 'uninhabited':
        continue

    #get the offset from UTC (offset 0)
    pyt = timezone(timezone_str).localize(datetime(2016, 8, 1, 0, 00, 00))

    #set the offset in the table
    country_state_combos.set_value(i, 'offset', pyt.utcoffset().total_seconds())


#the only nulls, added manualy
country_state_combos = country_state_combos.set_value(99, 'offset', -10*60*60)
country_state_combos = country_state_combos.set_value(197, 'offset', 60*60)
country_state_combos = country_state_combos.set_value(267, 'offset', 13*60*60)
country_state_combos = country_state_combos.set_value(316, 'offset', 12*60*60)
country_state_combos = country_state_combos.set_value(323, 'offset', 11*60*60)
country_state_combos = country_state_combos.set_value(362, 'offset', 9*60*60)
country_state_combos = country_state_combos.set_value(636, 'offset', -4*60*60)




events_parsing = events_parsing.merge(country_state_combos, how='left', on = ['country','state']).drop(events_parsing.columns[[2,3,4]], axis=1)


#merge with display_ids that have a null geo_location
events = events.merge(events_parsing.drop('timestamp', axis = 1), how = 'left', on = 'display_id').drop('geo_location',axis = 1)


#some of the offsets are nulls since there are some corrupt geo_locations
events = events.fillna(0)


#add offset to timestamp as miliseconds
events.timestamp = events.timestamp + (events.offset * 1000)


#if timestamp is negative add a weeks' time
events.loc[events['timestamp'] < 0, 'timestamp'] = events['timestamp'] + 1000*60*60*24*7


#converting timestamp field to hour and day
events["hour"] = (events.timestamp // (3600 * 1000)) % 24
events["day"] = events.timestamp // (3600 * 24 * 1000)


#adding weekend column
#diving day hours to 4 sections to combine results of adjacent hours
events["weekend"] = events["morning"] = events["noon"] = events["evening"] = events["night"] = 0
events.ix[events['day'].isin([4, 5, 11, 12]), 'weekend'] = 1
events.ix[(events["hour"] < 6) | (events["hour"] > 23), "night"] = 1
events.ix[(events["hour"] >= 6) & (events["hour"] < 12), "morning"] = 1
events.ix[(events["hour"] >= 12) & (events["hour"] < 18), "noon"] = 1
events.ix[(events["hour"] >= 18) & (events["hour"] <= 23), "evening"] = 1


events.drop(['timestamp', 'hour', 'day','offset'], axis=1, inplace=True)


events.to_csv(tables_path + 'time_table.csv', index = False)


print ("time_table.csv created in " + tables_path + " directory")
