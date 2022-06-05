import pandas as pd
import numpy as np
from dateutil import parser
import datetime
import os
# test = pd.read_pickle('With_Features/Meteo/imbalance_2019Daily')

# balance data contains every minute originally and price /////imbalance data contains imbalances every 15 originally and reg state
#Add Datetime column
def add_datetime(df, source = 'tennet', type = 'balance'):
    if source == 'tennet':
        if type == 'balance':
            time_array = np.array(df["TIME"])
        elif type == 'imbalance':
            time_array = np.array(df["PERIOD_FROM"])
    else:
        time_array = np.array(df["ENTSOE"])

    date_array = np.array(df["DATE"])
    zipped = zip(time_array, date_array)
    datetime = [" ".join(i) for i in zipped]
    dfnew = df
    dfnew["DATETIME"] = datetime
    parsed = [parser.parse(t) for t in df["DATETIME"]]
    df["PARSED_DATETIME"] = parsed

    return dfnew
#Add Datetime
def AddParsedDateTimeToFile(infile, filepathend, source = 'tennet', type = 'balance'):
    orig = pd.read_pickle(infile)
    withdt = add_datetime(orig, source, type)
    withdt.to_pickle('With_DateTime/'+ filepathend)
    print('pickle created at', ' With_DateTime/', str(filepathend))

def add_weekend(df):
    DoW = df.index.to_series().dt.dayofweek
    df["Day_of_Week"] = DoW
    weekday = df.index.to_series().dt.weekday
    weekendbool = [1 if w>4 else 0 for w in weekday]
    df["Weekend"] = weekendbool
    return df
def add_daynight(df):
    # hours = df.index.hour
    # df['hour'] = hours
    daylist = []
    for time in df.index.time:
        if (time > datetime.time(8, 30, 00))   &  (time < datetime.time(23, 30, 00)):
            daylist.append("0")
        else:
            daylist.append("1")

    df['nighttime'] = daylist

    return df

def add_half_hour(df):
    hhlist = []
    hlist = []
    for time in df.index.time:
        if time.minute < 30:
            hhlist.append(time.hour)
        else:
            hhlist.append(time.hour + 0.5)
        hlist.append(time.hour)
    df['halfhour'] = hhlist
    df['hour'] = hlist

    return df

def add_season(df, year = 2019):
    seasonlist = []
    for date in df.index:
        if (date > datetime.datetime(year, 3, 19))   &  (date < datetime.datetime(year, 6, 21)): #20th - 20th inclusive #00.00 on 20-3 is inexplicably Winter
            seasonlist.append("Spring")
        elif (date > datetime.datetime(year, 6, 20))   &  (date < datetime.datetime(year, 9, 22)):
            seasonlist.append("Summer")
        elif (date > datetime.datetime(year, 9, 20))   &  (date < datetime.datetime(year, 12, 22)):
            seasonlist.append("Autumn")
        else:
            seasonlist.append("Winter")

    df['season_cat'] = seasonlist
    df["season"] = pd.Categorical(df["season_cat"]).codes

    return df
def create_meteodf(csv, granularity = '15T'): #at the moment it picks the length to correspond with tthe 1 year data
    pd.read_csv(csv)
    meteodata = pd.read_csv(csv)
    parsed = [parser.parse(t) for t in meteodata["datetime"]]
    meteodata["PARSED_DATETIME"] = parsed
    meteo = meteodata.set_index('PARSED_DATETIME')
    meteoresampled = meteo.resample(granularity).pad() #T is minutes
    meteoresampled = meteoresampled[:35136]

    return meteoresampled
def add_meteo(origdf, meteodf):
    # l = len(meteodf)
    energy = origdf#[:l]
    combineddf = pd.concat([energy, meteodf], axis=1)
    return combineddf


def clean(sampleperiod = '15T', balance = 'balance_2019', year = 2019, resamplemethod = 'ffill()' ):

    #Resample - Another Theoretical decision for later

    resampleinfile = ['With_DateTime/', balance] #['With_DateTime/', 'balance_2019']
    dtinfile = pd.read_pickle(''.join(resampleinfile))
    resampled = dtinfile.resample(sampleperiod, on='PARSED_DATETIME').mean()
    dayinfo = add_weekend(resampled)
    hh = add_half_hour(dayinfo)
    nightinfo = add_daynight(hh)
    seasonal = add_season(nightinfo, year)
    seasonal['Time'] = seasonal.index.tolist()

    if sampleperiod == '15T':
        seasonal.to_pickle(''.join(['With_Features/', resampleinfile[1]]))   #,15minutes
        print("meteoless pickle created at orig location")
        test = pd.read_pickle(''.join(['With_Features/', resampleinfile[1]]))

    # elif sampleperiod != '15T':
    #
    #     outfile = ''.join(['With_Features/', sampleperiod, '/', resampleinfile[1], '_', sampleperiod])
    #     os.makedirs(os.path.dirname(outfile), exist_ok=True)
    #
    #     meteodata = pd.read_csv('Meteorological Data/Daily/Amsterdam2019Daily.csv')
    #     meteodata["PARSED_DATETIME"] = [parser.parse(t) for t in meteodata["datetime"]]
    #
    #     if sampleperiod in ['5T', '1T', '3T']:
    #         resampledmeteo = meteodata.resample(sampleperiod, on='PARSED_DATETIME').mean()
    #
    #     else: #sampleperiod in ['240T', '180T', '360T', '1D', '2D', '4D', '7D', '14D']:
    #         resampledmeteo = meteodata.set_index("PARSED_DATETIME").resample(sampleperiod).pad()
    #         resampledmeteo = resampledmeteo[:len(seasonal)]
    #
    #     outdf = pd.concat([seasonal, resampledmeteo], axis = 1)
    #     outdf.to_pickle(outfile)
    #     print("pickle created at sampleperiod location")
    #     test = pd.read_pickle(outfile)

    return test

def createresampledfeaturemeteodfs(times = ['1T', '60T', '180T', '240T', '360T', '1D', '2D', '7D', '14D']):
    if '15T' not in times:
        for i in times:
            phi = clean(i, 'imbalance_2019')
            psi = clean(i, 'balance_2019')
    else:
        print('error: dont resample 15T')
        phi = psi = 'error'

    return phi, psi
#
# d, e = createresampledfeaturemeteodfs(['30T'])

#TODO Used for the original files with no resampling, probably needs to be redone so it fits cleanly with the clean finction
def createmeteo(originaldf, meteocsv, mode ='Daily', balance = 'balance', year = '2019'):
    if mode == 'Daily':
        meteodf = create_meteodf(meteocsv)
        original = pd.read_pickle(originaldf)
        combined = add_meteo(original, meteodf)
        infile = 'With_Features/Meteo/' + balance + '_' + year + mode
        print(infile)
        if balance == 'balance':
            combined.to_pickle(infile)
        elif balance == 'imbalance':
            combined.to_pickle(infile)
    # if mode == 'hourly':
    #     meteodf = create_meteodf(meteocsv)
    #     original = pd.read_pickle(originaldf)
    #     combined = add_meteo(original, meteodf)
    #     combined.to_pickle('With_Features/Meteo/balance_2019Hourly')

    return combined

# clean('15T', 'balance_2019')
# clean('15T', 'imbalance_2019')
# x = createmeteo('With_Features/balance_2019', 'Meteorological Data/Daily/Amsterdam2019Daily.csv', 'daily', 'balance')
# a = createmeteo('With_Features/imbalance_2019', 'Meteorological Data/Daily/Amsterdam2019Daily.csv', 'daily', 'imbalance')



