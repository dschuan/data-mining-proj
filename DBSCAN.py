import processdata as procd
import importcsv as ic

if __name__=='__main__':
    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")