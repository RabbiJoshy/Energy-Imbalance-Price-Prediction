import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.cbook as cbook

data = pd.read_pickle("Per15/Imbalance_2019")
datab = data[:300]
datab.plot(y = "MID_PRICE", use_index=True)

chris = 'christmas'

# time_array = np.array(data["PARSED_DATETIME"])
# price_array = np.array(data["MID_PRICE"])


fig, ax = plt.subplots()
# common to all three:

# ax.plot('date', 'adj_close', data=data)
ax.plot(time_array, price_array)

    # Major ticks every half year, minor ticks every month,
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.grid(True)
ax.set_ylabel(r'Price [\$]')

ax.set_title('DefaultFormatter', loc='left', y=0.85, x=0.02, fontsize='medium')

plt.show()