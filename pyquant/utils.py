import copy

import pandas as pd

def merge_list(starting_list):
    final_list = []
    for i,v in enumerate(starting_list[:-1]):
        if set(v)&set(starting_list[i+1]):
            starting_list[i+1].extend(list(set(v) - set(starting_list[i+1])))
        else:
            final_list.append(v)
    final_list.append(starting_list[-1])
    return final_list

def findValleys(y, srt):
    peak = y.iloc[srt]
    for left in xrange(srt-1, -1, -1):
        val = y.iloc[left]
        if val == 0 or val > peak:
            break
    right = len(y)
    for right in xrange(srt+1, len(y)):
        val = y.iloc[right]
        if val == 0 or val > peak:
            break
    return left, right

def fit_theo_dist(params, ny, ty):
    right_limit, scaling = params
    index = xrange(len(ny)) if len(ny) > len(ty) else xrange(len(ty))
    exp_dist = pd.Series(0, index=index)
    theo_dist = pd.Series(0, index=index)
    exp_dist += pd.Series(ny)
    exp_dist[int(right_limit):] = 0
    theo_dist += ty
    return ((exp_dist-theo_dist*scaling)**2).sum()

def looper(selected=None, df=None, theo=None, index=0, out=None):
    if out is None:
        out = [0]*len(selected)
    if index != len(selected):
        for i in selected[index]:
            out[index] = i
            for j in looper(selected=selected, df=df, theo=theo, index=index+1, out=out):
                yield j
    else:
        vals = pd.Series([df[i] for i in out])
        vals = (vals/vals.max()).fillna(0)
        residual = ((theo-vals)**2).sum()
        yield (residual, copy.deepcopy(out))