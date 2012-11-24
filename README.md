batchtrain
==========

Find the best model, using random hyperparameter optimization, using scikit-learn.

Usage
-----

1. Create a starcluster cluster, perhaps using spot instances.
2. Preprocess your data in sklearn format. [describe more]
3. Run ./queue-scikit-jobs.py
    which will queue up a bunch of ./scikit-job.py jobs.

...

Requirements
------------

scikit-learn.

License
-------

Released under the 3-clause BSD license (see LICENSE).
