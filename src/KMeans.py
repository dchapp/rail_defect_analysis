from pyspark.mllib.clustering import KMeans, KMeansModel

"""
Arguments: An RDD of lists of strings (possibly mixed numerical and non-numerical data)
Output: An RDD of NumPy arrays of purely numerical data
"""
def kmeans_preprocess(data):
    ### Obtain the subset of the data consisting of only the purely numerical
    ### columns and only the rows that have entries for each such column. 
    ### First eliminate any incomplete rows
    complete_rows = data.filter(lambda x: '' not in x)

def kmeans_run(data, num_clusters, max_iters, ):
    clusters = KMeans.train(data, 
                            num_clusters, 
                            max_iters, 
                            runs=10, 
                            initializationMode="random")
    return clusters

def get_error_of_point(clusters, point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

def kmeans_evaluate(data, clusters):
    ### Compute Within-Set Sum of Squared Error WSSSE
    errors = data.map(lambda point: error(clusters, point))
    WSSSE = errors.reduce(lambda x,y: x+y)
    print("Within-Set Sum of Squared Error = " + str(WSSSE))
    return WSSSE

def kmeans_display(clusters):
    return 0
