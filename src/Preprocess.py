import numpy as np

"""
Accepts a "record" that should be a list of elements of arbitrary type.
"""
def get_numerical_subrecord(record):
    out = []
    for e in record:
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out

"""
"""
def get_pure_numerical_subset(rdd):
    return rdd.map(lambda record: get_numerical_subrecord(record))


