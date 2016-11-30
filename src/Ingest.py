from xlsx2csv import convert

class_label = { "track_geometry_defects":"Exception Type", "rail_defects":"Defect Type" }

def ingest(data_file):
    csv = convert(data_file)
    default_ingest(csv)

"""
All this does right now is accept a csv file 
and return an RDD.
"""
def default_ingest(csv_file):
    data = sc.parallelize(csv_file)
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



         
	

