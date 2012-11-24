#!/usr/bin/python
"""
Brute force learn the problem, using scikit.
WARNING: We convert ALL features to DENSE!
"""

import sys
import string
import simplejson
import re

import os

import random

# Don't do this, because we want a different randomization each time.
#random.seed(0)

from locals import *

from hyperparameters import all_hyperparameters, MODEL_HYPERPARAMETERS

if __name__ == "__main__":
    modelconfigs = []
    for model in MODELS_TO_USE:
        oldlen = len(modelconfigs)
        for h in all_hyperparameters(MODEL_HYPERPARAMETERS[model]):
            modelconfigs.append((model, h))
        print >> sys.stderr, "%d model configurations for %s" % (len(modelconfigs)-oldlen, model)
    random.shuffle(modelconfigs)
    print >> sys.stderr, "%d model configurations" % len(modelconfigs)

    files = 0
    cmds = []
    for i, modelconfig in enumerate(modelconfigs):
        model, h = modelconfig
        cmd = "./scikit-job.py --kfold --model %s --hyperparameters %s" % (model, repr(simplejson.dumps(h)))
        cmds.append(cmd)
        
        files += 1
        if files % JOBS_PER_FILE == 0:
            jobcmd = "job%06d.sh" % (files/JOBS_PER_FILE)
            jobfile = open(jobcmd, "wt")
            jobfile.write("#!/bin/sh\n")
            for cmd in cmds:
                jobfile.write(cmd + "\n")
            cmds = []
            os.system("chmod +x %s" % jobcmd)
            print "qsub -V -b y -cwd ./%s" % jobcmd
            sys.stdout.flush()
            
#        if i > 1000: break
