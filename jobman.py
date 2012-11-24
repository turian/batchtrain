#!/usr/bin/python
"""
Manage the output of job runs.
"""

import simplejson
import common.mongodb
import pymongo
import pymongo.objectid

c = common.mongodb.collection("jobman", "jobs", HOSTNAME="master")
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
