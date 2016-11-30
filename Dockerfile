FROM dchapp/sparkpandas

ADD ./src/driver.py /src/driver.py
ADD ./src/KMeans.py /src/KMeans.py
ADD ./src/SummaryStatistics.py /src/SummaryStatistics.py
ADD ./src/xlsx2csv.py /src/xlsx2csv.py
ADD ./src/DecisionTree.py /src/DecisionTree.py

ADD ./data/rail_defects.xlsx /data/rail_defects.xlsx
ADD ./data/rail_defects.csv /data/rail_defects.csv
ADD ./data/track_geometry_defects.xlsx /data/track_geometry_defects.xlsx 
ADD ./data/track_geometry_defects.csv /data/track_geometry_defects.csv 

ADD ./run_analysis.sh /run_analysis.sh
ADD ./README.md /README.md

