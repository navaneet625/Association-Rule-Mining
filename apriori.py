import time
from collections import Counter
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
#running apriori using inbuilt library
from apyori import apriori

def loaddata(fileName):
	with open(fileName) as f:
		content  = f.readlines()
	print("No of lines in data :",len(content))
	# content = [x.strip() for x in line]
	dataset = []
	for i in content:
		#data is separated by -1 ,so split data on the basis of -1
		line  = i.split(' -1 ')
		#checking last element ,if it is '-2\n' then remove it
		if line[len(line)-1] == '-2\n':
			line.pop() 
		dataset.append(line)
	return dataset

def finditemset(dataset):
	itemset = []
	for i in dataset:
		for q in i:
			if q not in itemset:
				itemset.append(q)
	return itemset

def findcandidateSet(itemset,dataset):
	candSet = Counter()
	for i in itemset:
		for data in dataset:
			if i in data:
				candSet[i]+=1
	return candSet

def findlset(candSet,minsupCount):
	lset = Counter()
	for i in candSet:
		if candSet[i]>=minsupCount:
			lset[frozenset([i])]+=candSet[i]
	return lset


def findnextItemSet(data,l,count):
	nc = set()
	length = len(l)
	for i in range(0,length):
		for j in range(i+1,length):
			t = l[i].union(l[j])
			if(len(t)==count):
				nc.add(t)
	return nc


def printData(itemset_list):
	for i in itemset_list:
		print(str(list(i))+": " +str(itemset_list[i]))
	print()

def findnextCandset(nextItemSet,dataset,lset,count):
	nextCandset = Counter()
	for i in nextItemSet:
		nextCandset[i]=0
		for data in dataset:
			temp = set(data)
			if(i.issubset(temp)):
				nextCandset[i]+=1
	return nextCandset

def findnextLset(candSet,minsupCount,count):
	lset = Counter()
	for i in candSet:
		if candSet[i]>=minsupCount:
			lset[i]+=candSet[i]
	return lset

def computeFreqItemSet(l,data,s):
	pl = l
	pos = 1
	for count in range (2,1000):
		nc = list(findnextItemSet(data,list(l),count))
		c = findnextCandset(nc,data,list(l),count)
		# print("C"+str(count)+" : ")
		# printData(c)
		l = findnextLset(c,s,count)
		# print("L"+str(count)+" : ")
		# printData(l)
		if(len(l) == 0):
			break

		pl = l
		pos = count
	return pl,pos 


def apriori_scratch(data,s):
	start = time.time()
	init=finditemset(data)
	print(init)
	c = findcandidateSet(init,data)
	# print("C1 : ")
	# printData(c)

	l = findlset(c,s)
	# print("L1 : ")
	# printData(l)

	finalFreqSet,freq_set_len = computeFreqItemSet(l,data,s)
	end = time.time()

	print("Frequent item Set : ")
	printData(finalFreqSet)
	print("Time Taken is:",end-start)


####improvement using partition ###############
def computeFreqItemSet_p(l,data,s,candset):
    pl = l
    pos = 1
    for count in range (2,1000):
        nc = list(findnextItemSet(data,list(l),count))
        c = findnextCandset(nc,data,list(l),count)
        l = findnextLset(c,s,count)
        if(len(l) == 0):
            break
        candset.extend(list(c))
        pl = l
        pos = count
    return candset 


def apriori_p(data,s):
    start = time.time()
    init=finditemset(data)
    # print(init)
    candset = findcandidateSet(init,data)
    l = findlset(candset,s)
    candset = list(candset)
    candset.extend(computeFreqItemSet_p(l,data,s,candset))
    return candset

def apriori_partitions(transactions,num_partitions,min_sup):
    lendata = len(transactions)
    parts=list()
    freqitemsets=list()
    start=0
    size_partition=int(lendata/num_partitions)
    time_list=list()

    start_t = time.time()
    for c in range(num_partitions-1):
        parts.append( transactions[start:start+size_partition] )
        start=start+size_partition

    if start<lendata:
        parts.append(transactions[start:])

    end_t = time.time()
    time_partitioning=end_t-start_t
    print("Time Required for partitioning :",time_partitioning)
    c=1
    for i in parts:
        print("Length of partition ",c," :",len(i))
        c=c+1
    c=1
    freqitemsets=list()
    for p in parts:
        s = time.time()
        candset=apriori_p(p,min_sup)

        freqitemsets.extend(candset)
        en = time.time()
        partition_time=en-s
        time_list.append(partition_time)
        print("Time required by partition number :",c," is:",partition_time)
        c=c+1
    freqitemsets = findcandidateSet(freqitemsets,transactions)
    lset = findlset(freqitemsets,min_sup)
    printData(lset)
    return lset


print("Enter file name : eg arm.txt")
fileName = input()

print("Enter minimum support : eg 2")
s = int(input())

data = loaddata(fileName)

# freq_items = apriori(data, min_support=0.001, use_colnames=True, verbose=1)

# apriori_scratch(data,s)

# apriori_partitions(data,3,s)