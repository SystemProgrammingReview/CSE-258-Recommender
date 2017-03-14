import time
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy
import urllib
import scipy.optimize

events = {}

# Data is from:
# http://www.bayareabikeshare.com/open-data

# Extract the time information from the events
for f in ["201508_trip_data.csv", "201608_trip_data.csv", "201408_babs_open_data/201408_trip_data.csv", "201402_babs_open_data/201402_trip_data.csv"]:
  f = open(f, 'r')
  l = f.readline()
  for l in f:
    l = l.split(',')
    tripID = l[0]
    timeString = l[2]
    timeUnix = time.mktime(datetime.datetime.strptime(timeString, "%m/%d/%Y %H:%M").timetuple())
    events[tripID] = [timeUnix, timeString]

# Find the earliest event
earliest = None
for event in events:
  if earliest == None or events[event][0] < earliest[0]:
    earliest = events[event]

earliestTime = earliest[0]

hourly = defaultdict(int)

# Count events by hour
for event in events:
  t = events[event][0]
  hour = int(t - earliestTime) // (60*60)
  hourly[hour] += 1

f = open("hourly.json", 'w')
f.write(str(dict(hourly)) + '\n')

hourly = eval(open("hourly.json").read())

# Observations sorted by hour
hourlySorted = []
for h in hourly:
  hourlySorted.append((h,hourly[h]))

hourlySorted.sort()

X = [x[0] for x in hourlySorted]
Y = [x[1] for x in hourlySorted]

# Plot the raw observation data
plt.plot(X,Y)
plt.show()


# Plot using a sliding window
sliding = []

wSize = 24*7

tSum = sum([x[0] for x in hourlySorted[:wSize]])
rSum = sum([x[1] for x in hourlySorted[:wSize]])

for i in range(wSize,len(hourlySorted)-1):
  tSum += hourlySorted[i][0] - hourlySorted[i-wSize][0]
  rSum += hourlySorted[i][1] - hourlySorted[i-wSize][1]
  sliding.append((tSum*1.0/wSize,rSum*1.0/wSize))

X = [x[0] for x in sliding]
Y = [x[1] for x in sliding]

plt.plot(X,Y)
plt.show()


# Autoregressive features
def feature(hour):
  previousHours = []
  for i in [1,2,3,4,5,24,24*7,24*7*365]:
    previousHour = hour - i
    previousHourExists = previousHour in hourly
    if previousHourExists:
      # Use the feature if it doesn't exist
      previousHours += [0, hourly[previousHour]]
    else:
      # Otherwise add a "missing value" indicator
      previousHours += [1, 0]
  return previousHours

X = [feature(x) for x in hourly]
y = [hourly[x] for x in hourly]

theta,residuals,rank,s = numpy.linalg.lstsq(X, y)
