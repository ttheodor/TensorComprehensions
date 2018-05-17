#!/bin/env python3

from itertools import groupby

with open('hashes') as f:
    data = f.read()

hashes = data.split('\n')

if not hashes[-1]:
    del hashes[-1]

data = [(k,v) for k,v in zip(hashes, range(len(hashes)))]

data = sorted(data, key=lambda x:x[0])

groups = []
uniquekeys = []

for k,g in groupby(data, lambda x:x[0]):
    groups.append(list(g))
    uniquekeys.append(k)

for g in groups:
    if len(g) == 1:
        continue
    ids = [i for h,i in g]
    print(ids)
