#!/bin/bash
spark-submit --master=local[*] /src/driver.py -s -f $1
