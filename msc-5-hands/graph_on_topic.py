# coding: utf-8

import os
import json
from datetime import datetime
from matplotlib import pyplot as plt

with open(os.path.expanduser("~/.config/on_topic/data.json"), 'r') as f:
    d = json.load(f)

def tonum(s):
    if s == 'yes':
        return 2
    elif s == 'no':
        return 1
    else:
        return 0

d = [(datetime.fromisoformat(dt), a) for dt, a in d]
d = [(dt.replace(tzinfo=None), a) for dt, a in d]

weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
d0 = d[0][0]
dx = [(dt.date() - d0.date()).days for dt, s in d]
maxday = max(dx)
ticks = list(range(0, maxday + 1))
labels = [weekdays[(d0.weekday() + i)%7] for i in ticks]
dxlabel=[dt.strftime('%a') for dt, a in d]

hours = ["0:00 AM", "1:00 AM", "2:00 AM", "3:00 AM", "4:00 AM", "5:00 AM", "6:00 AM", "7:00 AM", "8:00 AM", "9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM", "5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM", "10:00 PM", "11:00 PM"]
dy = [dt.time().hour + dt.time().minute / 60 for dt, a in d]
dyticks = [i for i in range(0, 24)]
dylabels = [hours[i] for i in dyticks]

colors = ['snow', 'rebeccapurple', 'gold']
dcol = [colors[tonum(a)] for dt, a in d]

plt.scatter(x=dx, y=dy, c=dcol)
plt.xticks(ticks=ticks, labels=labels)
plt.yticks(ticks=dyticks, labels=dylabels)
plt.gca().invert_yaxis()
plt.show()
