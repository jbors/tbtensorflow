import os
import csv

DIR = "./data/MontgomerySet/ClinicalReadings/"

csv_file = open(DIR+ 'results.csv', "w", newline='')
writer = csv.writer(csv_file, delimiter=',')
headers = ['Sex','Age','Diagnosis','Marker']
writer.writerow(headers)

for filename in os.listdir(DIR):
    if filename.endswith(".txt"):
        with open(DIR + filename) as f:
            content = f.readlines()
            normal = False
            data = []
            for line in content:
                if line.__contains__(":"):
                    line = line.split(":")[1]
                line = line.rstrip()
                line = line.strip()
                if line != "":
                    data.append(line)
                if(line == "normal"):
                    normal = True
            if normal:
                data.append('0')
            else:
                data.append('1')
            writer.writerow(data)
        continue
    else:
        continue

