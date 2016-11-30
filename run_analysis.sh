#!/bin/bash
spark-submit --master=local[*] /src/driver.py -s -f /data/track_geometry_defects.xlsx
