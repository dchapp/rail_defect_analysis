from pyspark.mllib.stat import Statistics

def summary_preprocess(data):
    return 0

def summary_run(data):
    ### colStats expects an RDD of "vectors" e.g. numpy arrays
    summary = Statistics.colStats(data)
    return summary

def summary_display(summary):
    return 0
