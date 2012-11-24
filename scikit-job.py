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

from sklearn.cross_validation import KFold

from optparse import OptionParser

from common.stats import stats

import random

import cPickle
import scipy.sparse
import time

# Force us to get a numerical result, ever if the job will fall below TASKMIN.
FORCE = False
#FORCE = True

# For each task, what is the minimum value for each k-fold that we should continue?
# We look at the top 7 best models for each task num, and then pick the min over the kfolds in those jobs.
TASKMIN = {3: 0.68, 4: 0.61, 5: 0.71, 7: 0.66, 9: 0.77, 11: 0.66, 13: 0.51, 14: 0.43, 15: 0.82}

def modelstr(ty, clf):
#    return simplejson.dumps(("%s" % clf.__class__, clf.get_params(), ty))
    return repr(("%s" % clf.__class__, clf.get_params(), ty))

#@timeout(600)
def _train(num, ty, clf, train_set_x, train_set_y, valid_set_x, valid_set_y, job, kfold=False):
    FOLDS = 5
    if kfold:
        # Combine, for k-fold validation
#        try:
#           # WARNING: This produces bizarre results on numpy arrays
#            X = scipy.sparse.vstack([train_set_x, valid_set_x]).tocsr()
#        except:
#            X = numpy.vstack([train_set_x, valid_set_x])
        assert type(train_set_x) in [numpy.matrixlib.defmatrix.matrix, numpy.ndarray]
        assert type(valid_set_x) in [numpy.matrixlib.defmatrix.matrix, numpy.ndarray]
        X = numpy.vstack([train_set_x, valid_set_x])
        Y = numpy.hstack([train_set_y, valid_set_y])
        kf = KFold(len(Y), FOLDS, indices=True)

    try:
            start = time.clock()
            print >> sys.stderr, "trying %d %s" % (num, modelstr(ty, clf))
            errs = []
            if kfold:
                for i, (train, test) in enumerate(kf):
                    X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
#                    print X_train.shape, y_train.shape
                    clf.fit(X_train, y_train)
                    y_test_predict = clf.predict(X_test)
                    errs.append(rsquared(y_test, y_test_predict))
                    print >> sys.stderr, "INTERMEDIATE", num, "kfold=%d/%d" % (i+1,FOLDS), errs[-1], modelstr(ty, clf)
                    print >> sys.stderr, stats()
                    if num in TASKMIN and errs[-1] < TASKMIN[num] and i+1 < FOLDS:
                        if FORCE:
                            print >> sys.stderr, "FORCE=True, otherwise we'd abort becase err %f < %d taskmin %f" % (errs[-1], num, TASKMIN[num])
                        else:
                            print >> sys.stderr, "ABORTING. err %f < %d taskmin %f" % (errs[-1], num, TASKMIN[num])
                            job.result = False
                            return
            else:
                clf.fit(train_set_x, train_set_y)
                pred_valid_y = clf.predict(valid_set_x)
                errs.append(rsquared(valid_set_y, pred_valid_y))

            end = time.clock()
            difftime = end - start
            if kfold:
                job.result = {"mean": numpy.mean(errs), "std": numpy.std(errs), "95conf": numpy.mean(errs) - 1.96*numpy.std(errs), "min": numpy.min(errs), "folds": errs, "time": difftime}
                print >> sys.stderr, num, "kfold=%d" % FOLDS, "mean", numpy.mean(errs), "std", numpy.std(errs), "95conf", numpy.mean(errs) - 1.96*numpy.std(errs), "min", numpy.min(errs), modelstr(ty, clf)
                print num, "kfold=%d" % FOLDS, "mean", numpy.mean(errs), "std", numpy.std(errs), "95conf", numpy.mean(errs) - 1.96*numpy.std(errs), "min", numpy.min(errs), modelstr(ty, clf)
            else:
                job.result = {"mean": numpy.mean(errs), "title": difftime}
                print num, numpy.mean(errs), modelstr(ty, clf)
            sys.stdout.flush()
            print >> sys.stderr, stats()
    except Exception, e:
            print >> sys.stderr, "Error %s %s on %s %s" % (type(e), e, num, modelstr(ty, clf))

def train(num, ty, clf, train_set_x, train_set_y, valid_set_x, valid_set_y, job, kfold=False):
    try:
        _train(num, ty, clf, train_set_x, train_set_y, valid_set_x, valid_set_y, job, kfold)
        assert job.result is not None
        print "JOB", job
        sys.stdout.flush()
    except Exception, e:
        print >> sys.stderr, "Error %s %s on %s %s" % (type(e), e, num, modelstr(ty, clf))

#def stack(x1, y1, x2, y2):
#    try:
#        X = scipy.sparse.vstack([x1,x2]).tocsr()
#    except:
#        X = numpy.vstack([x1,x2])
#    Y = numpy.hstack([y1,y2])
#    return X, Y


def runjob(model, h, num, features, kfold, job):
        # y-values
        valid_set_y, train_set_y = cPickle.load(open("datagz/sparse.test%d.pkl.gz" % num))

        valid_set_x, train_set_x, test_set_x = [None] * 3
        # Now load all features
        for f in features:
            vals = cPickle.load(open(f))
            new_valid_set_x, new_train_set_x, new_test_set_x = vals
            # Just features, no y-values
            try:
                new_valid_set_x, new_train_set_x, new_test_set_x = new_valid_set_x.todense(), new_train_set_x.todense(), new_test_set_x.todense()
            except:
                print >> sys.stderr, "%s already dense" % f

            if valid_set_x is None:
                valid_set_x, train_set_x, test_set_x = new_valid_set_x, new_train_set_x, new_test_set_x
            else:
                valid_set_x = numpy.hstack((valid_set_x, new_valid_set_x))
                train_set_x = numpy.hstack((train_set_x, new_train_set_x))
                test_set_x = numpy.hstack((test_set_x, new_test_set_x))
                
#            valid_set_x.append(new_valid_set_x)
#            train_set_x.append(new_train_set_x)
#            test_set_x.append(new_test_set_x)

#        for i in valid_set_x: print i.shape
#        valid_set_x = numpy.hstack(tuple(valid_set_x))
#        train_set_x = numpy.hstack(tuple(train_set_x))
#        test_set_x = numpy.hstack(tuple(test_set_x))

        print >> sys.stderr, "valid = %s, train = %s" % (valid_set_x.shape, train_set_x.shape)
        print >> sys.stderr, stats()

        clf = model(**h)
        train(num, features, clf, train_set_x, train_set_y, valid_set_x, valid_set_y, kfold=kfold, job=job)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--model", dest="model")
    parser.add_option("--hyperparameters", dest="hyperparameters")
    parser.add_option("--num", dest="num", type=int)
    parser.add_option("--kfold", dest="kfold", action="store_true", default=False)

    (options, args) = parser.parse_args()
#    print options.hyperparameters
    options.hyperparameters = simplejson.loads(options.hyperparameters)
    job = Job(model=options.model, hyperparameters=options.hyperparameters, num=options.num, features=args, kfold=options.kfold)
    if "cache_size" in job.parameters["hyperparameters"]:
        # This parameter shouldn't affect the result, it's just used to
        # determine training efficiency.
        del job.parameters["hyperparameters"]["cache_size"]
    if job.result is not None:
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
    runjob(model=options.model, h=options.hyperparameters, num=options.num, features=args, kfold=options.kfold, job=job)
