import pandas as pd

def convert(xlsx_file):
    print xlsx_file
    data_xlsx = pd.read_excel(xlsx_file, "Sheet1", index_col=None)
    csv_file = xlsx_file.split(".")[0] + ".csv"
    print "Converting " + xlsx_file + " to " + csv_file
    data_xlsx.to_csv(csv_file, encoding="utf-8")
    return csv_file

