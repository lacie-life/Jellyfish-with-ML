import json
import os
import re
import csv
import platform

import matplotlib.pyplot as plt
import numpy as np

datapath = '/home/lacie/Github/ML-Learning/Course/Theory/GAN/AdvanTopic/SeqGAN/data/'
redditfile = 'reddit_jokes.json'
stupidfile = 'stupidstuff.json'
wockafile = 'wocka.json'
outfile = 'jokes.csv'
headers = ['row', 'Joke', 'Title', 'Body', 'ID',
            'Score', 'Category', 'Other', 'Source']


def clean_str(text):
    fileters = '"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n\r\"\''
    trans_map = str.maketrans(fileters, " " * len(fileters))
    text = text.translate(trans_map)
    re.sub(r'[^a-zA-Z,. ]+', '', text)
    return text

def get_data(fn):
    with open(fn, 'r') as f:
        extracted = json.load(f)
    return extracted

def handle_reddit(rawdata, startcount):
    global writer
    print(f'Reddit file has {len(rawdata)} items...')
    cntr = startcount
    with open(outfile, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
    
    for d in rawdata:
        title = clean_str(d['title'])
        body = clean_str(d['body'])
        id = d['id']
        score = d['score']
        category = ''
        other = ''
        dict = {}
        dict['row'] = cntr
        dict['Joke'] = title + ' ' + body
        dict['Title'] = title
        dict['Body'] = body
        dict['ID'] = id
        dict['Category'] = category
        dict['Score'] = score
        dict['Other'] = other
        dict['Source'] = 'Reddit'
        writer.writerow(dict)
        cntr += 1
        if cntr % 10000 == 0:
            print(cntr)
    return cntr

def handle_stupidstuff(rawdata, startcount):
    global writer
    print(f'StupidStuff file has {len(rawdata)} items...')
    with open(outfile, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        cntr = startcount
        for d in rawdata:
            body = clean_str(d['body'])
            id = d['id']
            score = d['rating']
            category = d['category']
            other = ''
            dict = {}
            dict['row'] = cntr
            dict['Joke'] = body
            dict['Title'] = ''
            dict['Body'] = body
            dict['ID'] = id
            dict['Category'] = category
            dict['Score'] = score
            dict['Other'] = other
            dict['Source'] = 'StupidStuff'
            writer.writerow(dict)
            cntr += 1
            if cntr % 1000 == 0:
                print(cntr)
    return cntr
    
def handle_wocka(rawdata, startcount):
    global writer
    print(f'Wocka file has {len(rawdata)} items...')
    with open(outfile, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        cntr = startcount
        for d in rawdata:
            other = clean_str(d['title'])
            title = ''
            body = clean_str(d['body'])
            id = d['id']
            category = d['category']
            score = ''
            other = ''
            dict = {}
            dict['row'] = cntr
            dict['Joke'] = body
            dict['Title'] = title
            dict['Body'] = body
            dict['ID'] = id
            dict['Category'] = category
            dict['Score'] = score
            dict['Other'] = other
            dict['Source'] = 'Wocka'
            writer.writerow(dict)
            cntr += 1
            if cntr % 1000 == 0:
                print(cntr)
    return cntr

def prep_CVS():
    global writer
    with open(outfile, mode='a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()

def main():
    pv = platform.python_version()
    print(f"Running under Python {pv}")
    path1 = os.getcwd()
    print(path1)
    prep_CVS()
    print('Dealing with Reddit file')
    extracted = get_data(datapath + "/" + redditfile)
    count = handle_reddit(extracted, 0)
    print('Dealing with StupidStuff file')
    extracted = get_data(datapath + "/" + stupidfile)
    count = handle_stupidstuff(extracted, count)
    print('Dealing with Wocka file')
    extracted = get_data(datapath + "/" + wockafile)
    count = handle_wocka(extracted, count)
    print(f'Finished processing! Total items processed: {count}')

if __name__ == '__main__':
    main()


# fout = open("jokes.csv", "w")
# fout.write("id,text")
# fout.write('\n')
# id = 0
# lens = []

# for filename in sorted(["reddit_jokes.json"]):
#     with open(os.path.join('data', filename), mode='r') as fin:
#         jokes = json.load(fin)
#         for x in jokes:
#             t = x.get("title").strip()
#             s = x.get("body").strip()
#             s = t+'. '+s
#             s = clean_str(s)
#             l = len(s.split())
#             if l > 30:
#                 continue
#             lens.append(l)
#             fout.write("\"{}\",\"{}\"".format(id, s))
#             fout.write('\n')
#             id += 1
#             # if id > 100:
#             #     quit()
# for filename in sorted(["wocka.json", "stupidstuff.json"]):
#     with open(os.path.join('data', filename), mode='r') as fin:
#         jokes = json.load(fin)
#         for x in jokes:
#             s = x.get("body").strip().replace('\n', ' ')
#             s = s.replace('\"', '')
#             s = clean_str(s)
#             l = len(s.split())
#             if l > 30:
#                 continue
#             lens.append(l)
#             fout.write("{},{}".format(id, s))
#             fout.write('\n')
#             id += 1

# lens = np.array(lens)

# # plt.hist(lens, bins=20)
# # plt.show()

# fout.close()

# print(id)
