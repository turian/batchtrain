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
    # First command-line param should be the X_Y Pickle file.
    assert len(sys.argv) == 2
    X_Y_file = sys.argv[1]

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
    batchfile = "./output/submit-batch.sh"
    batch = open(batchfile, "wt")
    batch.write("#!/bin/sh\n")
    os.system("chmod +x %s" % batchfile)
    for i, modelconfig in enumerate(modelconfigs):
        model, h = modelconfig
        cmd = "./scikit-job.py --kfold --model %s --hyperparameters %s %s" % (model, repr(simplejson.dumps(h)), X_Y_file)
        cmds.append(cmd)
        
        files += 1
        if files % JOBS_PER_FILE == 0:
            jobcmd = "output/job%06d.sh" % (files/JOBS_PER_FILE)
            jobfile = open(jobcmd, "wt")
            jobfile.write("#!/bin/sh\n")
            for cmd in cmds:
                jobfile.write(cmd + "\n")
            cmds = []
            os.system("chmod +x %s" % jobcmd)
            batch.write("qsub -V -b y -cwd ./%s\n" % jobcmd)
#        if i > 1000: break
    print >> sys.stderr, "To submit the newly created jobs, run:"
    print >> sys.stderr, batchfile
