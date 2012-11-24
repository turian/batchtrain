#!/usr/bin/python

import jobman
import pymongo
from common.defaultordereddict import DefaultOrderedDict
from common.mydict import sort as dictsort
from common.str import percent
import numpy

import copy

TOP = 7         # Look at the top 7 jobs
MORE = 25       # And compare to the top 20 jobs

import re
pretrain_lr_re = re.compile("pretrain_lr=([0-9\.]+)\.")
layer_re = re.compile("/layer=([0-9]+)/")

def _inc(cnts, k, k2):
    if k2 not in cnts[k]: cnts[k][k2] = 0
    cnts[k][k2] += 1

def aggregate(jobs):
    cnts = DefaultOrderedDict(dict)
    for j in jobs:
        _inc(cnts, "hyperparameters.regressor", j["parameters"]["regressor"])
        for h in j["parameters"]["hyperparameters"]:
            _inc(cnts, "hyperparameters.%s" % h, j["parameters"]["hyperparameters"][h])
        #_inc(cnts, "hyperparameters.features", repr(j["parameters"]["features"]))
        m = pretrain_lr_re.search(repr(j["parameters"]["features"]))
        if m: _inc(cnts, "hyperparameters.features_pretrain_lr", m.group(1))
        for f in j["parameters"]["features"]:
            m = layer_re.search(f)
            if m: _inc(cnts, "hyperparameters.feature_layer", m.group(1))
        for f in j["parameters"]["features"]:
            _inc(cnts, "hyperparameters.feature", f)
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

def topjobs(num, CNT, verbose):
    prevj = None
    cnt = 0
    numjobs = []
    mins = []
    jobman.c.ensure_index([('parameters.num', 1)])
    if verbose:
        print "Task %s job results = %d" % (num, jobman.c.find({"parameters.num": num}).count())
    jobman.c.ensure_index([('parameters.num', 1), ('parameters.kfold', 1)])
    for j in jobman.c.find({"parameters.num": num, "parameters.kfold": True, "result": {"$ne": False}}).sort("result.95conf", pymongo.DESCENDING):
        if prevj is not None and prevj["parameters"] == j["parameters"]: continue # Skip duplicates
#        # Filter out a certain hyperpam value
#        if repr(j["parameters"]["features"]).find("pretrain_lr=0.1") == -1: continue
#        if j["parameters"]["hyperparameters"]["shrinking"] == False: continue
#        if j["parameters"]["hyperparameters"]["C"] != 100: continue
#        if j["parameters"]["regressor"] == "SVR": continue
#        if j["parameters"]["regressor"] != "SVR": continue
#        if j["parameters"]["hyperparameters"]["kernel"] != "RBF": continue
        if "time" in j["result"]: time = "time=%.2fm" % (j["result"]["time"]/60)
        else: time = "        "
        if verbose:
            print num, "%s mean=%.3f 95conf=%.3f min=%.3f" % (time, j["result"]["mean"], j["result"]["95conf"],j["result"]["min"]), j["parameters"]
        numjobs.append(j)
        mins.append(j["result"]["min"])
        prevj = j
        cnt += 1
        if cnt >= CNT: break
    if len(numjobs) < CNT:
        print "WARNING: Have %d < %d jobs for task %d" % (len(numjobs), CNT, num)
    if verbose:
        if len(mins) > 0:
            print "Task %d min(mins) = %f" % (num, numpy.min(mins))
    return numjobs

if __name__ == "__main__":
    alljobs = []
    alljobsmore = []
    bestvals = []
    print "Total job results = %d" % jobman.c.count()
    for num in range(1, 15+1):
        numjobs = topjobs(num, TOP, verbose=True)
        numjobsmore = topjobs(num, MORE, verbose=False)
        alljobs += numjobs
        alljobsmore += numjobsmore

        if len(numjobs) > 0:
            bestvals.append(numjobs[-1]["result"]["mean"])

        cnt = aggregate(numjobs)
        cntmore = aggregate(numjobsmore)
#        print_aggregate(cnt)
        print_aggregate_compare(cnt, cntmore)
        print
        print
    print
    cnt = aggregate(alljobs)
    cntmore = aggregate(alljobsmore)
#    print_aggregate(cnt)
    print_aggregate_compare(cnt, cntmore)

    print
    print
    print "OVERALL SCORE: %.4f (over %d tasks)" % (numpy.mean(bestvals), len(bestvals))
