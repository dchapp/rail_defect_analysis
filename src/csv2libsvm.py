#!/usr/bin/env python

"""
Convert CSV file to libsvm format. Works only with numeric variables.
Put -1 as label index (argv[3]) if there are no labels in your file.
Expecting no headers. If present, headers can be skipped with argv[4] == 1.
"""

import sys
import csv
from collections import defaultdict

def construct_line( label, line ):
    new_line = []
    if float( label ) == 0.0:
        label = "0"
    new_line.append( label )

    for i, item in enumerate( line ):
        if item == '' or float( item ) == 0.0:
            continue
        new_item = "%s:%s" % ( i + 1, item )
        new_line.append( new_item )
    new_line = " ".join( new_line )
    new_line += "\n"
    return new_line


def convert( csv_file, libsvm_file, target_col_index, has_header ):

    input_file = csv_file
    output_file = libsvm_file

    try:
        label_index = int( target_col_index )
    except IndexError:
        label_index = 0

    try:
        skip_headers = has_header
    except IndexError:
        skip_headers = 0

    i = open( input_file, 'rb' )
    o = open( output_file, 'wb' )

    reader = csv.reader( i )

    if skip_headers:
        headers = reader.next()

    for line in reader:
        if label_index == -1:
            label = '1'
        else:
            label = line.pop( label_index )

        new_line = construct_line( label, line )
        o.write( new_line )

    return libsvm_file

def main():
    convert(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

if __name__ == "__main__":
    main();
