#!/usr/bin/python

import jobman
import pymongo
from common.defaultordereddict import DefaultOrderedDict
from common.mydict import sort as dictsort
from common.str import percent
import numpy

import copy

#TOP = 7         # Look at the top 7 jobs
#MORE = 25       # And compare to the top 20 jobs
TOP = 25         # Look at the top 7 jobs
MORE = 100       # And compare to the top 20 jobs

def _inc(cnts, k, k2):
    if k2 not in cnts[k]: cnts[k][k2] = 0
    cnts[k][k2] += 1

def aggregate(jobs):
    cnts = DefaultOrderedDict(dict)
    for j in jobs:
        _inc(cnts, "hyperparameters.model", j["parameters"]["model"])
        for h in j["parameters"]["hyperparameters"]:
            _inc(cnts, "hyperparameters.%s" % h, j["parameters"]["hyperparameters"][h])
    return cnts

def print_aggregate(cnts):
    for k in cnts:
        print k
        tot = 0
        for v, k2 in dictsort(cnts[k]): tot += v
        for v, k2 in dictsort(cnts[k]):
            print "\t", percent(v, tot), k2

def print_aggregate_compare(cnts, cntsmore):
    """
    Compare the hyperparams in the TOP jobs to the hyperparams in the MORE jobs.
    """
    cntscopy = copy.deepcopy(cnts)
    for k in cnts:
        print k
        for k2 in cnts[k].keys():
            cntscopy[k][k2] = (1. * cnt[k][k2]/cntsmore[k][k2], cnts[k][k2], cntsmore[k][k2])
        maxperc = dictsort(cntscopy[k])[0][0][0]
        for v, k2 in dictsort(cntscopy[k]):
            # The second column (v[0]/maxperc) is a score for how good this hyperparam is.
            print "\t", k2, "\t", "%.2f" % (v[0]/maxperc), "\t", percent(v[1], v[2], rev=True)

def topjobs(CNT, verbose):
    prevj = None
    cnt = 0
    jobs = []
    mins = []
    if verbose:
        print "job results = %d" % (jobman.c.count())
    jobman.c.ensure_index([('parameters.kfold', 1)])
    for j in jobman.c.find({"parameters.kfold": True, "result": {"$ne": False}}).sort("result.95conf", pymongo.DESCENDING):
        if prevj is not None and prevj["parameters"] == j["parameters"]: continue # Skip duplicates
#        # Filter out a certain hyperpam value
#        if j["parameters"]["hyperparameters"]["shrinking"] == False: continue
#        if j["parameters"]["hyperparameters"]["C"] != 100: continue
#        if j["parameters"]["model"] != "SVR": continue
#        if j["parameters"]["hyperparameters"]["kernel"] != "RBF": continue
        if "time" in j["result"]: time = "time=%.2fm" % (j["result"]["time"]/60)
        else: time = "        "
        if verbose:
            print "%s mean=%.3f 95conf=%.3f min=%.3f" % (time, j["result"]["mean"], j["result"]["95conf"],j["result"]["min"]), j["parameters"]
        jobs.append(j)
        mins.append(j["result"]["min"])
        prevj = j
        cnt += 1
        if cnt >= CNT: break
    if len(jobs) < CNT:
        print "WARNING: Have %d < %d jobs" % (len(jobs), CNT)
    if verbose:
        if len(mins) > 0:
            print "min(mins) = %f" % (numpy.min(mins))
    return jobs

if __name__ == "__main__":
    bestvals = []
    print "Total job results = %d" % jobman.c.count()

    jobs = topjobs(TOP, verbose=True)
    jobsmore = topjobs(MORE, verbose=False)

    if len(jobs) > 0:
        bestvals.append(jobs[-1]["result"]["mean"])

    cnt = aggregate(jobs)
    cntmore = aggregate(jobsmore)
#    print_aggregate(cnt)
    print_aggregate_compare(cnt, cntmore)
    print
    print
