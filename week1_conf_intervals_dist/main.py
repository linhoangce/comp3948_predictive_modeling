import numpy as np
import matplotlib.pyplot as plt

years       = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
colorado    = [5029196,5029316,5048281,5121771,5193721,5270482,5351218,5452107,
               5540921,5615902,5695564]
connecticut = [3574097,3574147,3579125,3588023,3594395,3594915,3594783,3587509,
               3578674,3573880,3572665]
delaware = [897934,897934,899595,907316,915188,923638,932596,941413,949216,
               957078,967171]


plt.plot(years, colorado, '--', color='red', label='Colorado')
plt.plot(years, connecticut, 's', color='blue', label='Connecticut')
plt.plot(years, delaware, '-', color='green', label='Delaware')
plt.ylim(ymin=0) # set y axis to start at 0
plt.legend(loc='center left')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population by State')
plt.show()