from CleanerUtils import *

# AddParsedDateTimeToFile('Energy Data/imbalance-20190101-20200101.pickle', 'imbalance_2019', 'tennet', 'imbalance')
# AddParsedDateTimeToFile('Energy Data/imbalance-20180101-20190101.pickle', 'imbalance_2019', 'tennet', 'imbalance')
# AddParsedDateTimeToFile('Energy Data/balance_delta-20190101-20200101.pickle', 'balance_2019', 'tennet', 'balance')
# AddParsedDateTimeToFile('Energy Data/balance_delta-20180101-20190101.pickle', 'balance_2019', 'tennet', 'balance')

# d, e = createresampledfeaturemeteodfs(['30T'])

u = clean('15T', 'balance_2018', 2018)
uu = clean('15T', 'imbalance_2018', 2018)
uuu = clean('15T', 'balance_2019')
uuuu = clean('15T', 'imbalance_2019')

x = createmeteo('With_Features/balance_2018', 'Meteorological Data/Daily/Amsterdam2018Daily.csv', 'Daily', 'balance', '2018')
a = createmeteo('With_Features/imbalance_2018', 'Meteorological Data/Daily/Amsterdam2018Daily.csv', 'Daily', 'imbalance', '2018')
d = createmeteo('With_Features/balance_2019', 'Meteorological Data/Daily/Amsterdam2019Daily.csv', 'Daily', 'balance', '2019')
b = createmeteo('With_Features/imbalance_2019', 'Meteorological Data/Daily/Amsterdam2019Daily.csv', 'Daily', 'imbalance', '2019')

#  USe better variable names