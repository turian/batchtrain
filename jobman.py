#!/usr/bin/python
"""
Manage the output of job runs.
"""

import simplejson
import pymongo
import pymongo.objectid

from locals import EXPERIMENT_NAME

connection = pymongo.Connection('master', 27017)    # Use the master node for the DB
db = connection['jobman']
c = db[EXPERIMENT_NAME]
c.ensure_index('parameters', unique=True)

class Job(object):
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def get_result(self):
        r = c.find_one({"parameters": self.parameters})
        if r: return r["result"]
        else: return None

    def set_result(self, result):
        r = c.find_one({"parameters": self.parameters})
        if r:
            r["result"] = result
            c.save(r)
        else:
            c.save({"parameters": self.parameters, "result": result})

    result = property(get_result, set_result)

    def __str__(self):
        return simplejson.dumps({"parameters": self.parameters, "result": self.result})
