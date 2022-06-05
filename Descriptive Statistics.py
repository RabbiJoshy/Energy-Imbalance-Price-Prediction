import pandas as pd
from Clustering import fillmv
from Clustering import remove_outlier
from StatsUtils import *
import seaborn as sns
plt.style.use('seaborn')

mode = 'p'
period = '60T'
if mode == 'r':
    balance = 'regstate'
    depend = 'REGULATION_STATE'
elif mode == 'p':
    balance = 'price'
    depend = 'MID_PRICE'


if balance == 'price':
    if period == '15T':
        raw = pd.read_pickle("With_Features/Meteo/balance_2019Daily")
        clean = remove_outlier(fillmv(raw), 'MID_PRICE', criterion=1.5)
    else:
        raw = pd.read_pickle('With_Features/' + period + '/balance_2019_' + period)
        clean = remove_outlier(fillmv(raw), 'MID_PRICE', criterion=1.5)
        clean = raw
if balance == 'regstate':
    raw = pd.read_pickle('With_Features/Meteo/imbalance_2019Daily')
    clean = remove_outlier(fillmv(raw), 'REGULATION_STATE', criterion=1.5)
#Fill before removing outlier
#STATS
# description = data.describe()
# info = data.info()
# Weekend = pricedata.groupby(['Weekend']).mean()

# PLOTS
# plotminmax(clean)

# Outlier Detection - Boxplot
if balance == 'regstate':
    for feature in ['SEQ_NR', 'Weekend', 'season_cat', 'icon', 'uvindex']: # only shows significance for final element dunno why
    # for feature in ['Weekend']:
        sigs = boxplots(feature, depend, clean)
        plt.show()

if balance == 'price':
    for feature in ['nighttime']:#['SEQ_NR', 'Weekend', 'season_cat', 'icon', 'uvindex']: # only shows significance for final element dunno why
        sigs = boxplots(feature, depend, clean, 'on') #off if using seq_nr
        plt.show()

# df = pd.DataFrame(sigs, columns=['pair', 'significance'])
# df.plot(x ='pair', y='significance', kind = 'bar')
# plt.xticks(rotation=25)





#CORRELATION ['SEQ_NR', 'IGCCCONTRIBUTION_UP', 'IGCCCONTRIBUTION_DOWN',
       # 'UPWARD_DISPATCH', 'DOWNWARD_DISPATCH', 'RESERVE_UPWARD_DISPATCH',
       # 'RESERVE_DOWNWARD_DISPATCH', 'INCIDENT_RESERVE_UP_INDICATOR',
       # 'INCIDENT_RESERVE_DOWN_INDICATOR', 'MID_PRICE', 'MIN_PRICE',
       # 'MAX_PRICE', 'Day_of_Week', 'Weekend', 'season_cat', 'season', 'Time',
       # 'name', 'datetime', 'tempmax', 'tempmin', 'temp', 'feelslikemax',
       # 'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
       # 'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust',
       # 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
       # 'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
       # 'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations',
       # 'z_price'],

def corrstats():

    allcorrelationvars = ['solarradiation', 'temp', depend, 'Weekend', 'cloudcover', 'dew', 'feelslikemin', 'tempmin', 'SEQ_NR']
    pricecorrelationvars = clean.columns #['solarradiation', 'temp', 'MID_PRICE', 'Weekend', 'cloudcover','winddir','UPWARD_DISPATCH' ]
    corrMatrix = clean[allcorrelationvars].corr()

    l = clean[pricecorrelationvars].corr().columns.get_loc(depend)

    # pricecorrMatrix = clean[pricecorrelationvars].corr().iloc[9:10, :]
    sorted = clean[pricecorrelationvars].corr().iloc[l:l+1, :].sort_values(ascending = False, by = depend, axis = 1)
    newsorted= sorted[sorted>0.1]#need to invert this matrix some how

    sns.heatmap(sorted.iloc[:,np.r_[:10, -14:-3]], annot=True, cmap="YlGnBu")
    plt.xticks(rotation = 45)
    plt.show()

    sns.heatmap(corrMatrix, annot=True, cmap="YlGnBu")

corrstats()