import os
import csv
import math

def interval_length_comp():
    res_file = open('cifar10_test_full.csv', 'r')
    res = csv.reader(res_file, delimiter=',')
    res_in_list = list(res)
    i = 1
    lb_fullpath = "cifar10_1000.csv"
    while i <= 1000:
        with open(lb_fullpath, 'a+', newline='') as write_obj:
            csv_writer = csv.writer(write_obj)
            csv_writer.writerow(res_in_list[i])
        i = i + 1
        
if __name__ == "__main__":
    interval_length_comp()