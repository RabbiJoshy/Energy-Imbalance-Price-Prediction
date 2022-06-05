from Clustering import *

#data = pd.read_pickle("With_Features/balance_2019")
period = '60T'
infile = 'With_Features/'+ period + '/balance_2019_' + period
data = pd.read_pickle(infile)
pricedata = data[['MID_PRICE', 'MIN_PRICE', 'Day_of_Week','Weekend', 'SEQ_NR']]
pricedata['Time'] = data.index.tolist()


# def plot(data, shape, dates, time = ('00:00', '23:59'), clusters=2, highlow = ('all',35), weekend = 2, iv, dv):
argv = [pricedata,
        None,
        ('2019-1-1', '2019-3-30'),
        ('00:00', '23:59'),
        2,
        ('low', 35),
        2,
        'SEQ_NR',
        "MID_PRICE"]

if __name__ == "__main__":
    plot(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5])