batchtrain
==========

Find the best model, using random hyperparameter optimization, using scikit-learn.

Overview
--------

Use an SGE cluster, we pick the best model.

`./queue-scikit-jobs.py` queues a bunch of jobs, each with a random
hyperparameter configuration. (See Bergstra et al, JMLR.)

When each job runs, it caches the result (evaluation measure) of
running with those particular hyperparameters. If these hyperparameters
have already been run, this is determined automagically.

Usage
-----

0. Edit locals.py as desired.
1. Create an SGE cluster. The easiest way to do this is to use
starcluster to create an EC2 cluster. (Use spot instances to save
a lot of money.)
    1.1 Edit your .starcluster/config and make sure that every node
    in the instance (including master) has mongo and pymongo
    installed:

        [plugin pkginstaller]
        SETUP_CLASS = starcluster.plugins.pkginstaller.PackageInstaller
        # list of apt-get installable packages
        PACKAGES = mongodb, python-pymongo

    You will also probably need to have the following line in your `/etc/mongodb.conf`:

        bind_ip = 'master'

    after which you should restart the Mongo server. (WARNING: This might
    create a Mongo-server that is globally internet readable+writeable!)

    1.2 Make sure to:
        easy_install scikit-learn
    on every cluster node, including master.

2. Preprocess your data in sklearn format. [describe more]
3. Run `./queue-scikit-jobs.py`, which will queue up a bunch of
`./scikit-job.py` jobs.
4. Run bestmodel to get the best model.

...

Requirements
------------

* scikit-learn
* starcluster [optional]

License
-------

Released under the 3-clause BSD license (see LICENSE).
