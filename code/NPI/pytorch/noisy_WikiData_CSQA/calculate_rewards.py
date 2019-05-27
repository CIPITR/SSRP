#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 18:45:22 2018

@author: amrita
"""
import os
import cPickle as pkl
import numpy as np
import sys
from collections import Counter
rewards = pkl.load(open(sys.argv[1]))
orig_data = None
if len(sys.argv)>3:
	orig_data_dir = sys.argv[3]
	orig_data = {}
	for d in os.listdir(orig_data_dir):
		d = pkl.load(open(orig_data_dir+'/'+d))
		for di in d:
			if any([x in di[11].lower() for x in sys.argv[4].split(',')]):
				if di[-1]:
					orig_data[di[15]] = di[-1]
total_num_ques = len(set(['.'.join(x.split('.')[:2]) for x in orig_data]))
#print total_num_ques
top_k = int(sys.argv[2])
rewards_agg = {}
all_rewards_agg = {}
for k,v in rewards.items():
    k1 = '.'.join(k.split('.')[:2])
    good_data = False
    if orig_data and k in orig_data and orig_data[k]:
	good_data = True
    #print len(v)
    #print [v[i][-1] for i in xrange(len(v))]
    #if any([v[i][-1] for i in xrange(len(v))]):
    #    good_data =True
    if k1 not in all_rewards_agg:
        all_rewards_agg[k1] = []
    for i in range(len(v)):
        v[i][1] = max(0.0,v[i][1])
    all_rewards_agg[k1].extend(v)
    if not good_data:
	continue
    if k1 not in rewards_agg:
        rewards_agg[k1] = []
    for i in range(len(v)):
	v[i][-1] = True
	v[i][1] = max(0.0,v[i][1])	
    rewards_agg[k1].extend(v)
#print len(rewards_agg) 
for k in rewards_agg:
    rewards_agg[k] = sorted(rewards_agg[k], key=lambda x: x[0], reverse=True)
for k in all_rewards_agg:
    all_rewards_agg[k] = sorted(all_rewards_agg[k], key=lambda x: x[0], reverse=True)
rewards_new =[max(0,max([x[1] for x in v[:top_k]])) for v in rewards_agg.values() if v[0][-1]]
all_rewards_new =[max(0,max([x[1] for x in v[:top_k]])) for v in all_rewards_agg.values() if v[0][-1]]
#questions = list(set(['.'.join(k.split('.')[:2]) for k in rewards]))
total_num_ques = len(all_rewards_agg)
#print 'questions ', sorted(questions), len(questions)
print sys.argv[1].replace('results.pkl',''),'::: Avg. F1 of ',sum(rewards_new)/float(len(rewards_agg)), 'over ',len(rewards_agg), ' answerable questions (out of total', len(set(['.'.join(k.split('.')[:2]) for k in rewards])), 'questions'
#total_num_ques = len(set(['.'.join(k.split('.')[:2]) for k in rewards]))
#print 'Avg. F1 of ',sum(all_rewards_new)/float(len(all_rewards_agg)), 'over ',len(all_rewards_agg), ' questions (out of total', len(set(['.'.join(k.split('.')[:2]) for k in rewards])), 'questions'
