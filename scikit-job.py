#!/usr/bin/python
"""
Brute force learn the problem, using scikit.
WARNING: We convert ALL features to DENSE!
"""

import sys
import simplejson

from hyperparameters import MODEL_NAME_TO_CLASS

from jobman import Job

import numpy

from sklearn.cross_validation import KFold, LeaveOneOut
from sklearn.multiclass import OneVsRestClassifier
import sklearn.metrics

from optparse import OptionParser

from common.stats import stats

import random

import cPickle
import scipy.sparse
import time

# TASKMIN optimization disabled.
## Force us to get a numerical result, ever if the job will fall below TASKMIN.
#FORCE = False
##FORCE = True
## For each task, what is the minimum value for each k-fold that we should continue?
#TASKMIN = None

def modelstr(clf):
#    return simplejson.dumps(("%s" % clf.__class__, clf.get_params()))
    return repr(("%s" % clf.__class__, clf.get_params()))

#@timeout(600)
def train(model, h, X, Y, job, kfold):
    # TODO: These should be passed in as command-line parameters
    FOLDS = 5
    #FOLDS = 3
    EVALUATION_MEASURE = sklearn.metrics.f1_score

    if kfold: kf = KFold(X.shape[0], FOLDS, indices=True)
    #if kfold: kf = LeaveOneOut(X.shape[0], indices=True)
    else: assert 0

    start = time.clock()
    print >> sys.stderr, "trying %s %s" % (model, h)
    errs = []
    if kfold:
        for i, (train, test) in enumerate(kf):
            X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]

            clf = model(**h)
            # TODO: What we should do is have a multiclass command-line parameter,
            # in which case we do the following:
            clf = OneVsRestClassifier(clf)

            clf.fit(X_train, y_train)

            # TODO: Run evals on train, for debugging?

#            for j in range(y_test.shape[0]):
#                probs = []
#                for k, est in enumerate(clf.estimators_):
#                    print est.predict(X_test[j])
#                    y_test_predict = est.predict_proba(X_test[j])
#                    probs.append((y_test_predict[0][1], k))
#                    print y_test_predict[0][1]
#                print sorted(probs)[-1][0], y_test[j]

            y_test_predict = clf.predict(X_test)
            errs.append(EVALUATION_MEASURE(y_test, y_test_predict))
            print >> sys.stderr, "INTERMEDIATE kfold=%d/%d" % (i+1,FOLDS), errs[-1], modelstr(clf)
            print >> sys.stderr, stats()

#            if errs[-1] < TASKMIN and i+1 < FOLDS:
#                if FORCE:
#                    print >> sys.stderr, "FORCE=True, otherwise we'd abort becase err %f < %d taskmin %f" % (errs[-1], TASKMIN)
#                else:
#                    print >> sys.stderr, "ABORTING. err %f < %d taskmin %f" % (errs[-1], TASKMIN)
#                    job.result = False
#                    return
    else:
        assert 0

    end = time.clock()
    difftime = end - start
    if kfold:
        job.result = {"mean": numpy.mean(errs), "std": numpy.std(errs), "95conf": numpy.mean(errs) - 1.96*numpy.std(errs), "min": numpy.min(errs), "folds": errs, "time": difftime}
        print >> sys.stderr, "kfold=%d" % FOLDS, "mean", numpy.mean(errs), "std", numpy.std(errs), "95conf", numpy.mean(errs) - 1.96*numpy.std(errs), "min", numpy.min(errs), modelstr(clf)
        print "kfold=%d" % FOLDS, "mean", numpy.mean(errs), "std", numpy.std(errs), "95conf", numpy.mean(errs) - 1.96*numpy.std(errs), "min", numpy.min(errs), modelstr(clf)
    else:
        assert 0
#        job.result = {"mean": numpy.mean(errs), "title": difftime}
#        print num, numpy.mean(errs), modelstr(clf)
    sys.stdout.flush()
    print >> sys.stderr, stats()

def runjob(model, h, datafile, kfold, job):
    X, Y = cPickle.load(open(datafile))

    # TODO: Is it possible to get around doing this?
    # e.g. determine based upon "model" ?
    # At the very least, this should be a command-line param
    from locals import CONVERT_TO_DENSE
    if CONVERT_TO_DENSE:
        X = X.todense()

    print >> sys.stderr, "X = %s, Y = %s" % (X.shape, Y.shape)
    print >> sys.stderr, stats()

    try:
        train(model, h, X, Y, job, kfold)
        assert job.result is not None
        print "JOB", job
        sys.stdout.flush()
    except Exception, e:
        print >> sys.stderr, "Error %s %s on %s" % (type(e), e, (model, h))

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--model", dest="model")
    parser.add_option("--hyperparameters", dest="hyperparameters")
    parser.add_option("--kfold", dest="kfold", action="store_true", default=False)

    (options, args) = parser.parse_args()
    assert len(args) == 1
    datafile = args[0]
#    print options.hyperparameters
    options.hyperparameters = simplejson.loads(options.hyperparameters)
    job = Job(model=options.model, hyperparameters=options.hyperparameters, kfold=options.kfold)
    if "cache_size" in job.parameters["hyperparameters"]:
        # This parameter shouldn't affect the result, it's just used to
        # determine training efficiency.
        del job.parameters["hyperparameters"]["cache_size"]
    if 0 and job.result is not None:
        # If the result is False (below TASKMIN) and we aren't forcing a numerical result
        if job.result == False and not FORCE:
            print >> sys.stderr, "We already have a result for %s: %s" % (job.parameters, job.result)
            sys.exit(0)
        elif job.result != False:
            print >> sys.stderr, "We already have a result for %s: %s" % (job.parameters, job.result)
            sys.exit(0)
        elif job.result == False and FORCE:
            print >> sys.stderr, "We have a result for %s (%s), but FORCE=True" % (job.parameters, job.result)
        else:
            assert 0

    options.model = MODEL_NAME_TO_CLASS[options.model]
    runjob(model=options.model, h=options.hyperparameters, datafile=datafile, kfold=options.kfold, job=job)
