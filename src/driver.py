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


#### Necessary for dataframe stuff
#from pyspark import SQLContext
#sql_context = SQLContext(sc)
#
#
#### Functions for computing descriptive statistics
#from SummaryStatistics import summary_preprocess, summary_run, summary_display
#
#### Functions for decision tree classifier
#from DecisionTree import dt_preprocess, dt_run, dt_display
#
#### Functions for clustering using K-Means
#from KMeans import kmeans_preprocess, kmeans_run, kmeans_evaluate, kmeans_display
#
#### A function to convert xlsx files to csv since PySpark's textFile function
#### works with csv. Get rid of this after upgrading to the MLLib data frame API
#from xlsx2csv import convert


"""
Classification Task:
============================================================================
|           Data Set          | defect class | MGT | cumul. MGT | milepost |
============================================================================
| rail_defects.xlsx           |       H      |  S  |     U      |    D     |
| track_geometry_defects.xslx |       E      |     |            |          |
============================================================================

MLlib Classifiers to apply:
- decision tree (defect class)  
- regression tree (MGT, cumulative MGT, and milepost)
- Logistic Regression
- SVM
"""

def ingest(data_file, class_label_col):
    l = int(class_label_col)
    csv = convert(data_file)
    rdd = sc.textFile(csv).map(lambda row: row.split(","))
    rdd = rdd.map(lambda row: ( row[1:l], row[:l]+row[l+1:]))
    df  = rdd.toDF(["label", "features"])
    return df



#def ingest(data_file):
#    csv = convert(data_file)
#    return default_ingest(csv)

"""
All this does right now is accept a csv file as input read in each row, split 
it on tabs, and emit an RDD of numpy arrays.
"""
def default_ingest(csv_file):
    data = sc.parallelize(csv_file)
    #data = sc.textFile(csv_file)
    return data


def discard_incomplete_rows(data):
    ### Helper function to detect rows with missing data
    def check_row(row):
        if '' in row:
            return False
        else:
            return True
    data = data.filter(lambda row: check_row(row))
    return data

"""
Should be run AFTER discarding incomplete rows
"""
def discard_non_numerical_columns(data):
    ### Helper function to return truncated rows
    def truncate_row(row):
        truncated = []
        for x in row:
            try: 
                truncated.append(float(x))
            # Actually catching the exception here is apparently v. expensive.
            except:
                pass
        return np.array(truncated)
    data = data.map(lambda row: truncate_row(row))
    return data


def get_summary_statistics(infile):
    data = ingest(infile)
    summary = summary_run(data)








def xlsx_to_csv(xlsx_file):
    print xlsx_file
    data_xlsx = pd.read_excel(xlsx_file, "Sheet1", index_col=None)
    csv_file = xlsx_file.split(".")[0] + ".csv"
    print "Converting " + xlsx_file + " to " + csv_file
    data_xlsx.to_csv(csv_file, encoding="utf-8")
    return csv_file

def get_unique_values_by_column(data):
    headers = data.collect()[0]
    ### Get unique values in each column and associate
    ### this list of values with a column label
    kvps = []
    for h in headers:
        idx = headers.index(h)
        col_data = data.map(lambda x: x[idx]).filter(lambda x: x != h)
        distinct_values = col_data.distinct()
        kvp = (idx, distinct_values.collect())
        kvps.append(kvp)
    return sc.parallelize(kvps)



def my_isnumeric(string):
    try:
        return float(string)
    except ValueError:
        pass

"""
This function takes an RDD of mixed data and 
makes the col_idx column purely numerical, 
storing the number of distinct values.
"""
def categorical_to_numerical(data, col_idx):
    ### Get the column
    col = data.map(lambda x: x[col_idx])
    ### If it's already numerical, exit early
    if my_isnumeric(col.collect()[0]):
        ### Make it a float
        data = data.map(lambda x: x[:col_idx] + [float(x[col_idx])] + x[(col_idx+1):])
        return data, 0
    else:
        ### Get the set of unique values
        unique = list(set(col.collect()))
        num_categories = len(unique)
        #print "Num unique values for column " + str(col_idx) + ": " + str(len(unique))
        ### Make a map of unique labels to numerical proxies
        category_map = { k:v for k,v in zip(unique, range(len(unique))) }
        ### Knit the column back in
        data = data.map(lambda x: x[:col_idx] + [category_map[x[col_idx]]] + x[(col_idx+1):])
        return data, num_categories


def exclude_date_and_defect_size(data):
    return data.map(lambda x: x[:8] + x[10:])

def exclude_incomplete_rows(data):
    def f(x):
        if '' in x:
            return None
        else:
            return x
    data = data.map(lambda x: f(x))
    data = data.filter(lambda x: x != None)
    return data

def get_class_indices(data):
    classes = data.map(lambda x: x[7]) 
    unique_labels = list(set(classes.collect()))
    #pprint.pprint(unique_labels)
    class_indices = { k:v for k,v in zip(unique_labels, range(len(unique_labels))) }
    #pprint.pprint(class_indices)
    return class_indices

def to_labeled_point(record):
    label = record[7] ### for rail defects dataset
    feature_vector = record[:7] + record[8:]
    return LabeledPoint(label, feature_vector) 

def decision_tree_preprocess(data, class_indices):

    num_classes = len(class_indices)
    cat_features_info = {}
    data = exclude_date_and_defect_size(data)
    num_cols = len(data.collect()[0])

    ### Separate features and class labels
    feature_vectors = data.map(lambda x: x[:7] + x[8:])
    class_indices = get_class_indices(data)
    class_labels = data.map(lambda x: class_indices[x[7]])
    num_features = len(feature_vectors.collect()[0])

    for i in xrange(num_features):
        feature_vectors, num_categories = categorical_to_numerical(feature_vectors, i)
        if num_categories > 0 and num_categories < 32: ### Max bin thing
            cat_features_info[i] = num_categories
    
    print "Mapping records to labeled points"
    #data = data.map(lambda x: to_labeled_point(x))
    data = class_labels.zip(feature_vectors)
    data = data.map(lambda x: LabeledPoint(x[0], x[1]))

    print "Splitting into training and testing data sets"
    (training, testing) = data.randomSplit([0.8, 0.2])

    print "Categorical info dict"
    pprint.pprint( cat_features_info )

    print "Train the decision tree model"
    #model = DecisionTree.trainClassifier(training, numClasses=num_classes, categoricalFeaturesInfo={}, impurity="gini", maxDepth=5, maxBins=32)
    model = DecisionTree.trainClassifier(training, numClasses=num_classes, categoricalFeaturesInfo=cat_features_info, impurity="gini", maxDepth=20, maxBins=100)
    print "Training complete"

    print "Test the decision tree model against the set-aside testing data"
    predictions = model.predict(testing.map(lambda x: x.features))
    labels_and_predictions = testing.map(lambda x: x.label).zip(predictions)
    test_error = labels_and_predictions.filter(lambda (v,p): v!=p).count() / float(testing.count())
    print "Test error: " + str(test_error)
    #print "Learned model: " 
    #print model.toDebugString()

    # Instantiate metrics object
    metrics = MulticlassMetrics(labels_and_predictions)
    cm = metrics.confusionMatrix()
    #print cm

    #### This should contain kvps of the form:
    #### (n, k) meaning feature with column index n 
    #### has k possible values (categories)
    #unique_values = get_unique_values_by_column(data)
    #num_unique = unique_values.map(lambda x: (x[0], len(x[1])))
    #cat_features_info = { (x[0], x[1]) for x in num_unique.collect() }

    #pprint.pprint(cat_features_info)
    
    ### dummy return
    return (0,0,0)
     

def to_date(date_string):
    x = date_string.split("/")
    return datetime.date(int(x[2]), int(x[0]), int(x[1]))



def convert_categorical_to_numerical(data):
    ### Define an RDD of kvps of the form:
    ### ( col_label, [ val_0, val_1, ..., val_n-1 ] )
    col_vals = get_unique_values_by_column(data)
    ### Do binning if number of unique vals exceeds threshold
    threshold = 10 # just fix a value for now
    #binned = col_vals.map(lambda x: conditional_bin(x, threshold))
    

def build_decision_tree_model(data, class_indices):
    ### Preprocess 
    (data, cat_features_info, num_classes) = decision_tree_preprocess(data, class_indices)
    ### Result should be an RDD of LabeledPoint

    exit()

    ### Split data into training and testing sets
    (training_data, test_data) = data.randomSplit([0.7, 0.3])

    ### Train the model
    ### Needs a categoricalFeaturesInfo map indicating which features
    ### are categorical.
    model = DecisionTree.trainClassifier(training_data, numClasses = num_classes, categoricalFeaturesInfo = cat_features_info, impurity = "gini", maxDepth = 5, maxBins = 32)


def exclude_joint_weld(data):
    return data.map(lambda x: x[:14] + x[15:])

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

"""
Takes a CSV
Returns its rows as list of strings
"""
def load_data(csv_file):
    with open(csv_file, "rb") as infile:
        data = infile.readlines()
    return data



def to_list_of_str(data):
    ### Split string into list of string
    data = data.map(lambda x: x.split(","))
    ### Exclude excel row number column
    data = data.map(lambda x: x[1:])
    ### Strip whitespace
    data = data.map(lambda x: [y.strip() for y in x])
    return data

def separate_header(data):
    tmp = data.collect()
    header = tmp[0]
    data = sc.parallelize(tmp[1:])
    return header, data


def get_divisions(data, header):
    ### Get the column index for the division
    div_idx = header.index(u'DIVISION')
    ### Map to KVP
    kvps = data.map(lambda x: (x[div_idx], x[:div_idx] + x[div_idx+1:]))
    ### Group by key
    divs = kvps.groupByKey()
    records_per_div = divs.map(lambda x: (x[0], len(x[1])))

    pprint.pprint(records_per_div.collect())
    return divs 

def get_subdivisions(data, header):
    ### Get the column index for the division
    div_idx = header.index(u'DIVISION')
    subdiv_idx = header.index(u'SUBDIVISION')
    ### Map to KVP
    kvps = data.map(lambda x: ((x[div_idx],x[subdiv_idx]), x[:subdiv_idx] + x[subdiv_idx+1:]))
    ### Group by key
    subdivs = kvps.groupByKey()
    records_per_subdiv = subdivs.map(lambda x: (x[0], len(x[1])))

    pprint.pprint(records_per_subdiv.collect())
    return subdivs 

def restrict_to_division(data, div):
    ### Group by key with division as key
    return 0

def restrict_to_subdivision(data, subdiv):
    return 0


def main():
    ### Parse command line arguments
    ap = argparse.ArgumentParser(description="Rail and Track Geometry Defect Analysis")
    ap.add_argument("--spark", "-s", action="store_true", default=False)
    ap.add_argument("--file", "-f", action="store", type=str)
    args = ap.parse_args()
   
    ### Load the data
    data_file = args.file
    data_file_name = "".join(data_file.split(".")[:-1])
    data_file_type = data_file.split(".")[-1]
    ### Check if csv file for target data set already exists
    csv_file = data_file_name + ".csv"
    if os.path.isfile(csv_file):
        if args.spark:
            data = sc.textFile(csv_file);
        else:
            data = load_data(csv_file)
    else:
        csv = xlsx_to_csv(data_file)
        if args.spark:
            data = sc.textFile(csv)
        else:
            data = load_data(csv_file)

    ### Basic preprocessing step
    data = to_list_of_str(data)

    ### Separate headers and data
    header, data = separate_header(data)

    print "Header: " 
    pprint.pprint(header)


    ### Subset by division, subdivision, etc. 
    divisions = get_divisions(data, header)
    subdivisions = get_subdivisions(data, header)

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
