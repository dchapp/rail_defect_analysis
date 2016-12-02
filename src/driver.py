import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sys
import time
import os 
import pprint 
import argparse

from pyspark import SparkContext, SparkConf
conf = SparkConf()
sc = SparkContext(conf=conf)

### For decision tree classification
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics


####################################################################################
##################################### IO  ##########################################
####################################################################################

"""
Data loading wrapper
"""
def load_data(data_file, args):
    data_file_name = "".join(data_file.split(".")[:-1])
    data_file_type = data_file.split(".")[-1]
    csv_file = data_file_name + ".csv"
    if os.path.isfile(csv_file):
        if args.spark:
            data = sc.textFile(csv_file);
        else:
            with open(csv, "rb") as infile:
                data = infile.readlines()
    else:
        csv = xlsx_to_csv(data_file)
        if args.spark:
            data = sc.textFile(csv)
        else:
            with open(csv, "rb") as infile:
                data = infile.readlines()
    return data 

"""
Converts xlsx file to csv using Pandas
"""
def xlsx_to_csv(xlsx_file):
    data_xlsx = pd.read_excel(xlsx_file, "Sheet1", index_col=None)
    csv_file = xlsx_file.split(".")[0] + ".csv"
    print "Converting " + xlsx_file + " to " + csv_file
    data_xlsx.to_csv(csv_file, encoding="utf-8")
    return csv_file

    


####################################################################################
################################ Preprocessing  ####################################
####################################################################################
"""
Assumes data is RDD of lists of 
cleaned strings.
"""
def separate_header(data):
    tmp = data.collect()
    header = tmp[0]
    data = sc.parallelize(tmp[1:])
    return header, data

"""
Assumes data comes directly from xlsx_to_csv,
specifically that each string in data has a
superfluous Excel row number as its first element.
The map splits of the delimeter, discards the 
first element, then strips the whitespace from 
each remaining element. 
"""
def to_list_of_str(data, delim=","):
    data = data.map(lambda x: x.split(delim))
    data = data.map(lambda x: x[1:])
    data = data.map(lambda x: [y.strip() for y in x])
    return data

"""
Assumes data is RDD of lists of 
cleaned strings
"""
def exclude_incomplete_rows(data):
    def f(x):
        if '' in x:
            return None
        else:
            return x
    data = data.map(lambda x: f(x))
    num_rows = len(data.collect())
    data = data.filter(lambda x: x != None)
    num_complete = len(data.collect())
    print "Number of rows: " + str(num_rows)
    print "Number of rows with no missing values: " + str(num_complete)
    print "Percentage remaining: " + str(float(num_complete)/num_rows)
    return data

####################################################################################
################################ Maps to subsets  ##################################
####################################################################################
"""
"""
def to_div(data, header, div):
    idx = header.index(u'DIVISION')
    return data.filter(lambda x: x[idx] == div)

def to_subdiv(data, header, div, subdiv):
    d_idx = header.index(u'DIVISION')
    s_idx = header.index(u'SUBDIVISION')
    return data.filter(lambda x: x[d_idx] == div and x[s_idx] == subdiv)

####################################################################################
################################ Maps to KVPS  #####################################
####################################################################################
"""
"""
def to_kvps_col_num_missing(data, header):
    kvps = []
    for i in xrange(len(header)):
        k = header[i]
        v = data.map(lambda x: x[i])
        num_elements = v.count()
        v = v.filter(lambda x: x != "")
        num_present = v.count()
        p = round(1 - float(num_present)/num_elements,2)
        #kvps.append((k, (num_present, num_elements-num_present)))
        kvps.append((k, p))
    return sc.parallelize(kvps)

"""
"""
def to_kvps_col_num_unique(data, header):
    kvps = []
    for i in xrange(len(header)):
        k = header[i]
        v = data.map(lambda x: x[i])
        uniques = v.distinct()
        num_unique = uniques.count()
        kvps.append((k, num_unique))
    return sc.parallelize(kvps)

"""
"""
def to_kvps_div_count(data, header):
    ### Get the column index for the division
    div_idx = header.index(u'DIVISION')
    ### Map to KVP
    kvps = data.map(lambda x: (x[div_idx], x))
    ### Group by key
    #divs = kvps.groupByKey()
    #records_per_div = divs.map(lambda x: (x[0], len(x[1])))
    #pprint.pprint(records_per_div.collect())
    records_per_div = kvps.countByKey()
    return records_per_div

"""
"""
def to_kvps_subdiv_count(data, header):
    ### Get the column index for the division
    div_idx = header.index(u'DIVISION')
    subdiv_idx = header.index(u'SUBDIVISION')
    ### Map to KVP
    kvps = data.map(lambda x: ((x[div_idx],x[subdiv_idx]), x))
    ### Group by key
    #subdivs = kvps.groupByKey()
    #records_per_subdiv = subdivs.map(lambda x: (x[0], len(x[1])))
    records_per_subdiv = kvps.countByKey()
    #pprint.pprint(records_per_subdiv.collect())
    return records_per_subdiv

"""
"""
def count_missing_values(row):
    cleaned = filter(lambda x: x != "", row)
    return len(row) - len(cleaned)

"""
"""
def to_kvps_row_num_missing(data):
    return data.map(lambda x: (x, count_missing_values(x)))

"""
"""
def to_class_labels(data, header, data_source):
    if data_source == "data/rail_defects.xlsx":
        idx = header[u"DEFECT TYPE"]
    elif data_source == "data/track_geometry_defects.xlsx":
        idx = header[u"EXCEPTION TYPE"]
    return data.map(lambda x: x[idx])


def to_col(data, i):
    return data.map(lambda x: x[i])

"""
"""
def build_col_value_to_idx_map(col_values):
    values = col_values.distinct().collect()
    return {k:v for k,v in zip(values,range(len(values)))}

####################################################################################
################################ Classifiers  ######################################
####################################################################################
"""
Convert to labeled point for classifiers
"""
def to_labeled_point(data_point, header, data_file):
    if data_file == "track_geometry":
        class_idx = header["EXCEPTION TYPE"]
    elif data_file == "rail":
        class_idx = header["DEFECT TYPE"]
    label = record[7] ### for rail defects dataset
    feature_vector = record[:7] + record[8:]
    return LabeledPoint(label, feature_vector) 

def restrict(x, header, class_label, feature_cols):
    cols = [class_label] + feature_cols
    ret = []
    for c in cols:
        ret.append(x[header[c]])
    return ret

def to_labeled_points(data, data_source, header, feature_cols=None):
    header = {k:v for k,v in zip(header,range(len(header)))}
    
    ### Restrict
    class_label = None
    if data_source == "data/rail_defects.xlsx":
        class_label = "DEFECT TYPE"
    elif data_source == "data/track_geometry_defects.xlsx":
        class_label = "EXCEPTION TYPE"
    data = data.map(lambda x: restrict(x, header, class_label, feature_cols))
    data = exclude_incomplete_rows(data)
    #print "Restricted data: "
    #pprint.pprint(data.collect()[:5])

    ### Rebuild header
    new_header = {k:v for k,v in zip([class_label]+feature_cols,range(len(feature_cols)+1))}
    #print "New header: "
    #print new_header

    ### Get feature vectors
    feature_vectors = data.map(lambda x: x[1:])
    #print "Feature vectors: "
    #pprint.pprint( feature_vectors.collect()[:5] )
    
    ### Get class labels
    class_labels = data.map(lambda x: x[0])
    #print "Class labels: " 
    #pprint.pprint( class_labels.collect()[:5] ) 
    
    ### Get class labels
    #class_labels = to_class_labels(data, header, data_source)
    #print "Original class labels: "
    #print "Num class labels: " + str(class_labels.count())
    #pprint.pprint( class_labels.collect()[:5] )
    class_label_to_idx_map = build_col_value_to_idx_map(class_labels)
    feature_value_to_idx_maps = {}
    for fc in feature_cols:
        ### Get the unique values in this feature's column
        col_values = to_col(data, new_header[fc])
        ### Build the map to translate the feature value labels into numeric labels
        feature_value_to_idx_maps[fc] = build_col_value_to_idx_map(col_values)
    feature_arities = []
    for fc in feature_cols:
        feature_arities.append(len(feature_value_to_idx_maps[fc]))
    catinfo = {k:v for k,v in zip(range(len(feature_cols)),feature_arities)}

    #### Get feature vectors
    #feature_vectors = data.map(lambda x: x[1:])
    #print "Feature vectors: "
    #pprint.pprint( feature_vectors.collect()[:5] )

    ### Map them to purely numerical representation
    class_labels = class_labels.map(lambda x: class_label_to_idx_map[x])
    #print "Class labels (converted): "
    #pprint.pprint( class_labels.collect()[:5] ) 
    def to_nums(x, feature_value_to_idx_maps, feature_cols):
        y = []
        for i in xrange(len(feature_cols)):
            y.append( feature_value_to_idx_maps[feature_cols[i]][x[i]] )
        return y
    feature_vectors = feature_vectors.map(lambda x: to_nums(x, feature_value_to_idx_maps, feature_cols))
    #print "Feature vectors (converted): "
    #pprint.pprint( feature_vectors.collect()[:5] )
    labels_and_feature_vectors = class_labels.zip(feature_vectors)
    #print "Joined class labels and feature vectors: "
    #pprint.pprint(labels_and_feature_vectors.collect()[:5])
    labeled_points = labels_and_feature_vectors.map(lambda x: LabeledPoint(x[0], x[1]))
    return labeled_points, catinfo



"""
Takes
Returns 
"""
def generate_longlat_plot_serial(data):
    ### Split string into list of string
    data = map(lambda x: x.split(","), data)
    ### Exclude the excel row number column
    data = map(lambda x: x[1:], data)
    ### String whitespace
    data = map(lambda x: [y.strip() for y in x], data)
    print "Rows before excluding rows with missing values: " + str(len(data))
    ### Exclude rows with missing columns
    data = filter(lambda x: "" not in x, data)
    print "Rows after excluding rows with missing values: " + str(len(data))
    ### Separate headers and data
    headers = data[0]
    data = data[1:]
    ### Convert longtiudes and lattitudes
    data = map(lambda x: [try_float(y) for y in x], data)
    ### Collect all unique labels
    labels = set()
    for x in data:
        labels.add(x[2])
    ### Build mapping of labels to ints
    labels = list(labels)
    label_map = {k:v for k,v in zip(labels,range(len(labels)))}
    ### Convert label strings 
    data = map(lambda x: [x[0],x[1],label_map[x[2]]], data)
    ### Get coordinate lists
    xcoords = map(lambda x: x[0], data)
    ycoords = map(lambda x: x[1], data)
    ### Generate the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcoords,ycoords)
    plt.show()

def generate_longlat_plot_spark(data):
    ### Split string into list of string
    data = data.map(lambda x: x.split(","))
    ### Exclude excel row number column
    data = data.map(lambda x: x[1:])
    ### Strip whitespace
    data = data.map(lambda x: [y.strip() for y in x])
    ### Exclude all columns except defect type, latitude, and longitude
    data = data.map(lambda x: [x[4],x[32],x[33]])
    ### Separate into headers and data
    tmp = data.collect()
    headers = tmp[0]
    data = sc.parallelize(tmp[1:])
    ### Exclude rows with missing values
    print "Rows before excluding rows with missing values: " + str(len(data.collect()))
    data = data.filter(lambda x: "" not in x)
    print "Rows after excluding rows with missing values: " + str(len(data.collect()))
    ### Convert latitude and longitude data
    data = data.map(lambda x: [try_float(y) for y in x])

    ### Write out just lat and lon data
    tmp = data.map(lambda x: [(str(x[0])+", "), (str(x[1])+", ") ,str(x[2])]).map(lambda x: "".join(x)).collect()
    with open("/src/latlong.csv", "wb") as outfile:
        for line in tmp:
            outfile.write(line+"\n")

    ### Convert to KVP
    data = data.map(lambda x: (x[0], (x[1],x[2])))
    ### Get number of each kind of defect
    defect_counts = data.groupByKey().map(lambda x: (x[0], len(x[1])))
    tmp = defect_counts.collect()
    pprint.pprint(tmp)

"""
Takes something
Returns float(something) if possible
"""
def try_float(x):
    try: 
        return float(x)
    except ValueError:
        return x

#"""
#Takes a CSV
#Returns its rows as list of strings
#"""
#def load_data(csv_file):
#    with open(csv_file, "rb") as infile:
#        data = infile.readlines()
#    return data



#def to_list_of_str(data):
#    ### Split string into list of string
#    data = data.map(lambda x: x.split(","))
#    ### Exclude excel row number column
#    data = data.map(lambda x: x[1:])
#    ### Strip whitespace
#    data = data.map(lambda x: [y.strip() for y in x])
#    return data

#def separate_header(data):
#    tmp = data.collect()
#    header = tmp[0]
#    data = sc.parallelize(tmp[1:])
#    return header, data



def plot_defect_histogram(data, data_source, header):
    if data_source == "data/rail_defects.xlsx":
        label = "DEFECT TYPE"
    elif data_source == "data/track_geometry_defects.xlsx":
        label = "EXCEPTION TYPE"
    defect_col = to_col(data, header.index(label))
    defect_col = defect_col.map(lambda x: (x, 1))
    counts = defect_col.reduceByKey(lambda a,b: a+b)
    pprint.pprint(counts.collect())

#def get_divisions(data, header):


def main():
    ### Parse command line arguments
    ap = argparse.ArgumentParser(description="Rail and Track Geometry Defect Analysis")
    ap.add_argument("--spark", "-s", action="store_true", default=False)
    ap.add_argument("--file", "-f", action="store", type=str)
    args = ap.parse_args()
   
    ### Load the data
    data_file = args.file
    print "Loading data from file: " + data_file
    data = load_data(data_file, args)

    ### Basic preprocessing step
    #delim = ","
    #data = to_list_of_str(data, delim)
    data = to_list_of_str(data)

    ### Separate headers and data
    header, data = separate_header(data)    
    print "Column Headers: " 
    pprint.pprint(header)
    

    ### Get the subset we want
    ### Subset by division, subdivision, etc. 
    num_per_div = to_kvps_div_count(data, header)
    num_per_subdiv = to_kvps_subdiv_count(data, header)
    complete_rows = exclude_incomplete_rows(data)

    #print "Number of data points per division: "
    #pprint.pprint(num_per_div.items())
    #print "Number of data points per subdivision: "
    #pprint.pprint(num_per_subdiv.items())

    ### Analysis of columns:
    ### How many missing values per column?
    ### How many unique values per column? 
    ### Plot a histogram on values in this column? 
    ### Descriptive statistics of values in this column? 
    col_num_missing = to_kvps_col_num_missing(data, header)
    col_num_unique = to_kvps_col_num_unique(data, header)
    col_metadata = col_num_unique.join(col_num_missing)

    #print "Column metadata: number of unique values, percentage of values missing"
    #pprint.pprint(col_metadata.collect())

    print "Defect Type Histogram: Whole Dataset"
    plot_defect_histogram(data, data_file, header)
   
    print "Defect Type Histogram: Complete rows only"
    plot_defect_histogram(complete_rows, data_file, header)


    print "Get the subset of the data for the (u'APPALACHIAN', u'BIG SANDY') subdivision"
    subdiv_key = (u'APPALACHIAN', u'BIG SANDY')
    subdiv_data = to_subdiv(data, header, subdiv_key[0], subdiv_key[1])
    print subdiv_data.count()
   
    print "Defect Type Histogram: APPALACHIAN - BIG SANDY subdivision only"
    plot_defect_histogram(subdiv_data, data_file, header)


    exit()
    
    ### Run classifier on subdivision-specific track-geometry defect data
    ### Use following features, treat all as categorical.
    ### 1. "CURVE"
    ### 2. "EVENT"
    ### 3. "FREIGHT_MPH_Q" 
    ### 4. "TRACK"
    feature_cols = ["CURVE",
                    "EVENT",
                    "FRIEGHT_MPH_Q",  ### Mis-spelling deliberate, it's spelled that way in the data
                    "TRACK"]
    labeled_points, catinfo = to_labeled_points(subdiv_data, data_file, header, feature_cols)
    #print catinfo
    #print labeled_points.collect()[:5]
    (training, testing) = labeled_points.randomSplit([0.7,0.3])
    num_classes = to_col(subdiv_data, header.index("EXCEPTION TYPE")).distinct().count()
    #print num_classes
    model = DecisionTree.trainClassifier(training, 
                                         numClasses=num_classes, 
                                         categoricalFeaturesInfo=catinfo,
                                         impurity="gini", 
                                         maxDepth=5, 
                                         maxBins=32)
    predictions = model.predict(testing.map(lambda x: x.features))
    labels_and_predictions = testing.map(lambda x: x.label).zip(predictions)
    test_error = labels_and_predictions.filter(lambda (v,p): v!=p).count() / float(testing.count())
    print "Test error: " + str(test_error)
    
    






    ##### Load the latitude-longitude track geometry
    ##### defect data. Produce a plot of the locations
    ##### colored by defect type.
    ##if args.spark:
    ##    generate_longlat_plot_spark(data)
    ##else:
    ##    generate_longlat_plot_serial(data)


    #data = data.map(lambda x: x.split(",")[1:])
    ##data = data.map(lambda record: [str(x) for x in record] ) ### de-unicodes
   
    #### Get rid of the ridiculously incomplete Joint-Weld column
    #data = exclude_joint_weld(data)

    #### Get rid of incomplete rows
    #print "Num records total: " + str(len(data.collect()))
    #data = exclude_incomplete_rows(data)
    #print "Num complete records: " + str(len(data.collect()))
   
    #tmp = data.collect()
    #headers = tmp[0]
    #data = sc.parallelize(tmp[1:])

    #class_indices = get_class_indices(data)

    #### Build decision tree and classify
    #build_decision_tree_model(data, class_indices)

if __name__ == "__main__":
    main()
