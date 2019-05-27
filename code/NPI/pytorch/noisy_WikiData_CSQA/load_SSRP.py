#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 14:28:29 2018

@author: amrita
"""

from model_SSRP import NPI
from read_data_testing import ReadBatchData
import itertools
from interpreter_testing import Interpreter
import numpy as np
import json
import random
import sys
import re
import string
import cPickle as pkl
import os
import math
from pipeproxy import proxy
from multiprocessing import Process, Lock, freeze_support
import msgpack
import time
import torch



class LoadModel():
    def __init__(self, param):
        np.random.seed(1)
        torch.manual_seed(999)
        #if torch.cuda.is_available(): torch.cuda.manual_seed_all(999)
        self.param = param
        self.run_interpreter = True
        self.run_validation = False
        self.generate_data = False
        self.param = param
        self.questype_program_dict = {'simple,logical':'gen_set','verify':'','simple':'gen_set', 'logical':'gen_set','quantitative':'gen_map1','quantitative count':'gen_set,gen_map1','comparative':'gen_set,gen_map1','comparative count':'gen_set,gen_map1'}
        if not self.generate_data and os.path.exists(self.param['model_dir']+'/model_data.pkl'):
            self.pickled_train_data = pkl.load(open(self.param['model_dir']+'/model_data.pkl'))
        else:
            self.pickled_train_data = {}
        self.starting_epoch = 0
        self.starting_overall_step_count = 0
        self.starting_validation_reward_overall = 0
        self.starting_validation_reward_topbeam = 0
        if 'dont_look_back_attention' not in self.param:
            self.param['dont_look_back_attention'] = False
        if 'concat_query_npistate' not in self.param:
            self.param['concat_query_npistate'] = False
        if 'query_attention' not in self.param:
            self.param['query_attention'] = False
        if self.param['dont_look_back_attention']:
            self.param['query_attention'] = True
        if 'single_reward_function' not in self.param:
            self.param['single_reward_function'] = False
        if 'terminate_prog' not in self.param:
            self.param['terminate_prog'] = False
            terminate_prog = False
        else:
            terminate_prog = self.param['terminate_prog']
        if 'dump_ert_train_data_freq' not in self.param:
            self.param['dump_ert_train_data_freq'] = 50    
        if 'none_decay' not in self.param:
            self.param['none_decay'] = 0
        if 'train_mode' not in self.param:
            self.param['train_mode'] = 'reinforce'
        self.qtype_wise_batching = self.param['questype_wise_batching']
        self.read_data = ReadBatchData(param)
        print "initialized read data"
        if 'quantitative' in self.param['question_type'] or 'comparative' in self.param['question_type']:
            if 'relaxed_reward_till_epoch' in self.param:
                relaxed_reward_till_epoch = self.param['relaxed_reward_till_epoch']
            else:
                self.param['relaxed_reward_till_epoch'] = [-1,-1]
                relaxed_reward_till_epoch = [-1,-1]
        else:
            self.param['relaxed_reward_till_epoch'] = [-1,-1]
            relaxed_reward_till_epoch = [-1,-1]
        if 'params_turn_on_after' not in self.param:
            self.param['params_turn_on_after'] = 'epoch'
        if self.param['params_turn_on_after']!='epoch' and self.param['params_turn_on_after']!='batch':
            raise Exception('params_turn_on_after should be epoch or batch')
        if 'print' in self.param:
            self.printing = self.param['print']
        else:
            self.param['print'] = False
            self.printing = True
        if 'prune_beam_type_mismatch' not in self.param:
            self.param['prune_beam_type_mismatch'] = 0
        if 'prune_after_epoch_no.' not in self.param:
            self.param['prune_after_epoch_no.'] = [self.param['max_epochs'],1000000]
        if self.param['question_type']=='verify':
            boolean_reward_multiplier = 1
        else:
            boolean_reward_multiplier = 0.1
        if 'print_valid_freq' not in self.param:
            self.param['print_valid_freq'] = self.param['print_train_freq']
        if 'valid_freq' not in self.param:
            self.param['valid_freq'] = 100
        if 'unused_var_penalize_after_epoch' not in self.param:
            self.param['unused_var_penalize_after_epoch'] =[self.param['max_epochs'],1000000]
        unused_var_penalize_after_epoch = self.param['unused_var_penalize_after_epoch']
        if 'epoch_for_feasible_program_at_last_step' not in self.param:
            self.param['epoch_for_feasible_program_at_last_step']=[self.param['max_epochs'],1000000]
        if 'epoch_for_biasing_program_sample_with_target' not in self.param:
            self.param['epoch_for_biasing_program_sample_with_target'] = [self.param['max_epochs'],1000000]
        if 'epoch_for_biasing_program_sample_with_last_variable' not in self.param:
            self.param['epoch_for_biasing_program_sample_with_last_variable'] = [self.param['max_epochs'],100000]
        if 'use_var_key_as_onehot' not in self.param:
            self.param['use_var_key_as_onehot'] = False
        reward_func = "f1"
        self.param['reward_function'] = reward_func
        if 'relaxed_reward_strict' not in self.param:
            relaxed_reward_strict = False
            self.param['relaxed_reward_strict'] = relaxed_reward_strict
        else:
            relaxed_reward_strict = self.param['relaxed_reward_strict']
        if param['parallel']==1:
            raise Exception('Need to fix the intermediate rewards for parallelly executing interpreter')
        for k,v in param.items():
            print 'PARAM: ', k , ':: ', v
        print 'loaded params '
        self.valid_data = []
        if os.path.isdir(param['valid_data_file']):
            self.valid_files = [param['valid_data_file']+'/'+x for x in os.listdir(param['valid_data_file']) if x.endswith('.pkl')]
        elif not isinstance(param['valid_data_file'], list):
            self.valid_files = [param['valid_data_file']]
        else:
            self.valid_files = param['valid_data_file']
	self.evaluation_data = {}
        for file in self.valid_files:
            d=pkl.load(open(file))
	    if len(d)==0:
		continue
            d = self.add_data_id(d)
	    d = self.remove_bad_data(d)
            self.valid_data.extend(d)
	print 'length of valid data', len(self.valid_data)
        random.shuffle(self.valid_data)    
        if self.qtype_wise_batching:
            self.valid_data_map = self.read_data.get_data_per_questype(self.valid_data)
            self.valid_batch_size_types = self.get_batch_size_per_type(self.valid_data_map)
            self.n_valid_batches = int(math.ceil(float(sum([len(x) for x in self.valid_data_map.values()])))/float(self.param['batch_size']))
        else:
            self.n_valid_batches = int(math.ceil(float(len(self.valid_data))/float(self.param['batch_size'])))

        if not os.path.exists(param['model_dir']):
            os.mkdir(param['model_dir'])
        self.model_file = os.path.join(param['model_dir'],param['model_file'])
        learning_rate = param['learning_rate']
        start = time.time()
        if torch.cuda.is_available():
            USE_DEVICE = 'cuda'
        else:
            USE_DEVICE = 'cpu'
        self.reward_beam_data = {}
        self.model = NPI(param, self.read_data.none_argtype_index, self.read_data.num_argtypes, \
                         self.read_data.num_progs, self.read_data.max_arguments, \
                         self.read_data.rel_index, self.read_data.type_index, \
                         self.read_data.wikidata_rel_embed, self.read_data.wikidata_type_embed, \
                         self.read_data.vocab_init_embed, self.read_data.program_to_argtype, \
                         self.read_data.program_to_targettype,USE_DEVICE)
        self.ert_train_data = {}
        self.checkpoint_prefix = os.path.join(param['model_dir'], param['model_file'])
        if os.path.exists(self.checkpoint_prefix):
            self.model.load_state_dict(torch.load(self.checkpoint_prefix))
            fr = open(self.param['model_dir']+'/metadata.txt').readlines()
            self.starting_epoch = int(fr[0].split(' ')[1].strip())
            self.starting_overall_step_count = int(fr[1].split(' ')[1].strip())
            print 'restored model'
        else:
            raise Exception('No model file exists')
        end = time.time()
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=[0.9,0.999], weight_decay=1e-5)
        print self.model

       
        print 'model created in ', (end-start), 'seconds'

        self.interpreter = Interpreter(self.param['wikidata_dir'], self.param['num_timesteps'], \
                                       self.read_data.program_type_vocab, self.read_data.argument_type_vocab, self.printing, terminate_prog, relaxed_reward_strict, reward_function = reward_func, boolean_reward_multiplier = boolean_reward_multiplier, relaxed_reward_till_epoch=relaxed_reward_till_epoch, unused_var_penalize_after_epoch=unused_var_penalize_after_epoch)
        if self.param['parallel'] == 1:
            self.InterpreterProxy, self.InterpreterProxyListener = proxy.createProxy(self.interpreter)
            self.interpreter.parallel = 1
            self.lock = Lock()
        print "initialized interpreter"

    def perform_full_test(self, epoch, overall_step_count):
        valid_reward = 0
        valid_reward_at0 = 0
	print 'num valid batches',self.n_valid_batches
        for i in xrange(self.n_valid_batches):
            train_batch_dict = self.get_batch(i, self.valid_data, self.valid_data_map, self.valid_batch_size_types)
            avg_batch_reward_at0, avg_batch_reward, _ = self.perform_test(train_batch_dict, epoch, overall_step_count)
            if i%self.param['print_valid_freq']==0 and i>0:
                valid_reward += avg_batch_reward
                valid_reward_at0 += avg_batch_reward_at0
                avg_valid_reward = float(valid_reward)/float(i+1)
                avg_valid_reward_at0 = float(valid_reward_at0)/float(i+1)
                print ('Valid Results in Epoch  %d Step %d (avg over batch) valid reward (over all) =%.6f valid reward (at top beam)=%.6f running avg valid reward (over all)=%.6f running avg valid reward (at top beam)=%.6f' %(epoch, i, avg_batch_reward, avg_batch_reward_at0, avg_valid_reward, avg_valid_reward_at0))
        overall_avg_valid_reward = valid_reward/float(self.n_valid_batches)
        overall_avg_valid_reward_at0 = valid_reward_at0/float(self.n_valid_batches)
        return overall_avg_valid_reward_at0, overall_avg_valid_reward

    def feeding_dict2(self, reward, IfPosIntermediateReward, mask_IntermediateReward, IntermediateReward, relaxed_rewards, mask):
        feed_dict = {}
        feed_dict['reward'] = reward
        feed_dict['IfPosIntermediateReward'] = IfPosIntermediateReward
        feed_dict['mask_IntermediateReward'] = mask_IntermediateReward
        feed_dict['IntermediateReward'] = IntermediateReward
        feed_dict['Relaxed_reward'] = relaxed_rewards
        feed_dict['mask_reinforce'] = mask
        return feed_dict

    def feeding_dict1(self, encoder_inputs_w2v, encoder_inputs_kb_emb, variable_mask, \
                      variable_embed, variable_atten, kb_attention, batch_response_type, \
                      batch_required_argtypes, feasible_program_at_last_step, bias_prog_sampling_with_target,\
                      bias_prog_sampling_with_last_variable, epoch_inv, epsilon, PruneNow):
        feed_dict = {}
        feed_dict['encoder_text_inputs_w2v'] = np.transpose(encoder_inputs_w2v)
        feed_dict['encoder_text_inputs_kb_emb'] = encoder_inputs_kb_emb
        feed_dict['preprocessed_var_mask_table'] = variable_mask
        feed_dict['preprocessed_var_emb_table'] = variable_embed
        feed_dict['kb_attention'] = kb_attention
        # in phase 1 we should sample only generative programs
        temp = np.zeros([self.param['batch_size'], self.read_data.num_progs], dtype = np.int64)
        for i in self.read_data.program_variable_declaration_phase:
            temp[:,i] = 1
        feed_dict['progs_phase_1'] = temp
        # in phase 2 we should not sample generative programs and  we can sample all other programs
        temp = np.zeros([self.param['batch_size'], self.read_data.num_progs], dtype = np.int64)
        for i in self.read_data.program_algorithm_phase:
            temp[:,i] = 1
        feed_dict['progs_phase_2'] = temp
        feed_dict['gold_target_type'] = batch_response_type.astype(np.int64)
        feed_dict['randomness_threshold_beam_search'] = epsilon
        feed_dict['DoPruning'] = PruneNow
        feed_dict['last_step_feasible_program'] = feasible_program_at_last_step*np.ones((1,1))
        feed_dict['bias_prog_sampling_with_last_variable'] = bias_prog_sampling_with_last_variable*np.ones((1,1))
        feed_dict['bias_prog_sampling_with_target'] = bias_prog_sampling_with_target*np.ones((1,1))
        feed_dict['required_argtypes'] = batch_required_argtypes
        feed_dict['relaxed_reward_multipler'] = epoch_inv*np.ones((1,1), dtype=np.float32)
        return feed_dict

    def map_multiply(self, arg):
        orig_shape = arg[0].shape
        arg0 = np.reshape(arg[0], (self.param['batch_size']*self.param['beam_size'], -1))
        arg1 = np.reshape(arg[1], (self.param['batch_size']*self.param['beam_size'],1))
        mul = np.reshape(np.multiply(arg0, arg1), orig_shape)
        return np.sum(mul,axis=(0,1))

    # Disable
    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout = sys.__stdout__

    def get_ert_from_a_seq(self, a_seq, variable_value_table, prepopulated_var_mask):
        ents = set([])
        rels = set([])
        types = set([])
        for i in xrange(self.param['num_timesteps']):
            if a_seq['program_type'][i]==self.read_data.program_type_vocab['gen_set']:
                for j in xrange(self.read_data.max_arguments):
                    row = a_seq['argument_type'][i][j]
                    col = a_seq['argument_table_index'][i][j]
                    if prepopulated_var_mask[row][col]==1 and isinstance(variable_value_table[row][col], list):
                        print variable_value_table
                        print prepopulated_var_mask
                        raise Exception('Something wrong with variable value table')
                    if row==self.read_data.ent_index and prepopulated_var_mask[row][col]==1:
                        ents.add(variable_value_table[row][col])
                    if row==self.read_data.rel_index and prepopulated_var_mask[row][col]==1:
                        rels.add(variable_value_table[row][col])
                    if row==self.read_data.type_index and prepopulated_var_mask[row][col]==1:
                        types.add(variable_value_table[row][col])   
        return ents, rels, types 
    
    def parallel_forward_pass_interpreter(self, batch_orig_context, a_seq, per_step_probs, \
                                 program_probabilities, variable_value_table, batch_response_entities, \
                                 batch_response_ints, batch_response_bools):

        Reward_From_Model = np.transpose(np.array(a_seq['Model_Reward_Flag']))

        keys = ['program_type','argument_type','target_type','target_table_index','argument_table_index','variable_value_table']
        old_seq = dict.fromkeys(['program_type','argument_type','target_type','target_table_index','argument_table_index'])
        for key in old_seq:
            old_seq[key] = np.array(a_seq[key]).tolist()
        new_a_seq = [[dict.fromkeys(keys) for beam_id in xrange(self.param['beam_size'])] \
                      for batch_id in xrange(self.param['batch_size'])]
        def asine(batch_id,beam_id,key):
            new_a_seq[batch_id][beam_id][key] = ['phi' for _ in xrange(self.param['num_timesteps'])]
        [[[asine(batch_id,beam_id,key) for key in keys] for beam_id in xrange(self.param['beam_size'])] \
                              for batch_id in xrange(self.param['batch_size'])]


        def handle_variable_value_table(key,beam_id,timestep,batch_id):
            if key is not 'variable_value_table':
                new_a_seq[batch_id][beam_id][key][timestep] = old_seq[key][beam_id][timestep][batch_id]
            else:
                new_a_seq[batch_id][beam_id][key] = variable_value_table[batch_id].tolist()


        [handle_variable_value_table(key,beam_id,timestep,batch_id) for (key,beam_id,timestep,batch_id) in list(itertools.product\
         (keys,xrange(self.param['beam_size']),xrange(self.param['num_timesteps']),xrange(self.param['batch_size'])))]

        def calculate_program_reward(shared_object, arg_f):
            shared_object.calculate_program_reward(arg_f)

        def parallel_fetch_interpreter(l, f, arg_f, shared_object):
                l.acquire()
                f(shared_object, arg_f)
                l.release()

        self.interpreter.rewards = [[None for beam_id in xrange(self.param['beam_size'])] \
                              for batch_id in xrange(self.param['batch_size'])]

        self.blockPrint()
        jobs = []
        for batch_id in xrange(self.param['batch_size']):
            for beam_id in xrange(self.param['beam_size']):

                args = (new_a_seq[batch_id][beam_id], \
                           batch_response_entities[batch_id], \
                           batch_response_ints[batch_id], \
                           batch_response_bools[batch_id],
                           beam_id,batch_id)

                arg_f = msgpack.packb(args ,use_bin_type=True)
                p = Process(target=parallel_fetch_interpreter, args=(self.lock, calculate_program_reward, arg_f, \
                                                                 self.InterpreterProxy))
                jobs.append(p)
                p.start()
                self.InterpreterProxyListener.listen()

        while True in set([job.is_alive() for job in jobs]):
            self.InterpreterProxyListener.listen()

        [job.join() for job in jobs if job.is_alive()]
        self.enablePrint()

        for batch_id in xrange(self.param['batch_size']):
            if self.printing:
                print 'batch id ', batch_id, ':: Query :: ', batch_orig_context[batch_id]
            for beam_id in xrange(self.param['beam_size']):
                if self.printing:
                    print 'beam id', beam_id
                    print 'per_step_probs',per_step_probs[batch_id,beam_id]
                    print 'product_per_step_prob', np.product(per_step_probs[batch_id,beam_id])
                    print 'per_step_programs [',
                new_a_seq_i = new_a_seq[batch_id][beam_id]
                for timestep in range(len(new_a_seq_i['program_type'])):
                    prog = new_a_seq_i['program_type'][timestep]
                    args = new_a_seq_i['argument_table_index'][timestep]
                    if self.printing:
                        print self.read_data.program_type_vocab_inv[prog]+'( '+','.join([str(\
                                       self.read_data.argument_type_vocab_inv[self.read_data.program_to_argtype[prog][arg]])+\
                                       '('+str(args[arg])+')' for arg in range(len(args))])+' )',
                if self.printing:
                    print ']'
        rewards = np.array(self.interpreter.rewards)
        if self.param['reward_from_model']:
            rewards = np.where(Reward_From_Model == 0, rewards, -1*np.ones_like(rewards))
        return rewards

    def forward_pass_interpreter(self, preprocessed_var_mask, all_predicted, batch_ids, batch_orig_context, a_seq, per_step_probs, \
                                 program_probabilities, variable_value_table, batch_response_entities, \
                                 batch_response_ints, batch_response_bools, epoch_number, overall_step_count):
        program_probabilities = program_probabilities.data.cpu().numpy()
	preprocessed_var_mask = np.asarray(preprocessed_var_mask).transpose((2,0,1))
        keys = ['program_type','argument_type','target_type','target_table_index','argument_table_index','variable_value_table']

        new_a_seq = [[dict.fromkeys(keys) for beam_id in xrange(self.param['beam_size'])] \
                      for batch_id in xrange(self.param['batch_size'])]
        def asine(batch_id,beam_id,key):
            new_a_seq[batch_id][beam_id][key] = ['phi' for _ in xrange(self.param['num_timesteps'])]
        [[[asine(batch_id,beam_id,key) for key in keys] for beam_id in xrange(self.param['beam_size'])] \
                              for batch_id in xrange(self.param['batch_size'])]


        def handle_variable_value_table(key,beam_id,timestep,batch_id):
            if key is not 'variable_value_table':
                new_a_seq[batch_id][beam_id][key][timestep] = a_seq[key][beam_id][timestep][batch_id].data.cpu().numpy()
            else:
                new_a_seq[batch_id][beam_id][key] = variable_value_table[batch_id]


        [handle_variable_value_table(key,beam_id,timestep,batch_id) for (key,beam_id,timestep,batch_id) in list(itertools.product\
         (keys,xrange(self.param['beam_size']),xrange(self.param['num_timesteps']),xrange(self.param['batch_size'])))]

        for batch_id in xrange(self.param['batch_size']):
            for beam_id in xrange(self.param['beam_size']):
                new_a_seq[batch_id][beam_id]['program_probability'] = program_probabilities[batch_id][beam_id]


        rewards = []
        intermediate_rewards_flag = []
        mask_intermediate_rewards = []
        intermediate_rewards = []
        relaxed_rewards = []
        for batch_id in xrange(self.param['batch_size']):
            data_id = batch_ids[batch_id]    
            all_predicted_i = all_predicted[batch_id]
            if self.printing:
                print 'batch id ', batch_id, ':: Query :: ', batch_orig_context[batch_id]
            rewards_batch = []
            intermediate_rewards_flag_batch = []
            relaxed_rewards_batch = []
            mask_intermediate_rewards_batch = []
            intermediate_rewards_batch = []
            for beam_id in xrange(self.param['beam_size']):
                if self.printing:
                    print 'beam id', beam_id
                    print 'per_step_probs',per_step_probs[batch_id,beam_id]
                    print 'product_per_step_prob', np.product(per_step_probs[batch_id,beam_id])
                    print 'per_step_programs [',
                new_a_seq_i = new_a_seq[batch_id][beam_id]
                for timestep in range(len(new_a_seq_i['program_type'])):
                    prog = int(new_a_seq_i['program_type'][timestep])
                    args = new_a_seq_i['argument_table_index'][timestep]
                    if self.printing:
                        print self.read_data.program_type_vocab_inv[prog]+'( '+','.join([str(\
                                   self.read_data.argument_type_vocab_inv[self.read_data.program_to_argtype[prog][arg]])+\
                                   '('+str(args[arg])+')' for arg in range(len(args))])+' )',
                if self.printing:
                    print ']'
                args = (new_a_seq[batch_id][beam_id], \
                       batch_response_entities[batch_id], \
                       batch_response_ints[batch_id], \
                       batch_response_bools[batch_id])
                reward, max_intermediate_reward, relaxed_reward, intermediate_mask, intermediate_reward_flag = self.interpreter.calculate_program_reward(args, epoch_number, overall_step_count)
                if reward>0:
                    if data_id not in self.ert_train_data:
                        self.ert_train_data[data_id] = {}
                    if 'reward' not in self.ert_train_data[data_id] or reward > self.ert_train_data[data_id]['reward']:
                        ents, rels, types = self.get_ert_from_a_seq(new_a_seq[batch_id][beam_id], variable_value_table[batch_id], preprocessed_var_mask[batch_id])
                        d = {'reward':reward, 'entities':list(ents), 'relations':list(rels), 'types':list(types)} #add batch id here
                        self.ert_train_data[data_id] = d
                original_data_id = '.'.join(data_id.split('.')[:3])      
                if original_data_id not in self.evaluation_data:
                    self.evaluation_data[original_data_id] = []
                #program_probabilities is of dimension batch_size x beam_size
                #reward is of dimension batch_size x beam_size
                prog_prob_i = program_probabilities[batch_id][beam_id]
                self.evaluation_data[original_data_id].append([prog_prob_i, reward, all_predicted_i])    
                rewards_batch.append(reward)
                intermediate_rewards_flag_batch.append(intermediate_reward_flag)
                relaxed_rewards_batch.append(relaxed_reward)
                mask_intermediate_rewards_batch.append(intermediate_mask)
                intermediate_rewards_batch.append(max_intermediate_reward)

            rewards.append(rewards_batch)
            intermediate_rewards_flag.append(intermediate_rewards_flag_batch)
            mask_intermediate_rewards.append(mask_intermediate_rewards_batch)
            intermediate_rewards.append(intermediate_rewards_batch)
            relaxed_rewards.append(relaxed_rewards_batch)
        rewards = np.array(rewards)
        intermediate_rewards = np.array(intermediate_rewards)
        intermediate_rewards_flag = np.array(intermediate_rewards_flag)
        mask_intermediate_rewards = np.array(mask_intermediate_rewards)
        relaxed_rewards = np.array(relaxed_rewards)
        return rewards, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, mask_intermediate_rewards

    def get_ml_rewards(self,rewards):
        ml_rewards = np.zeros((self.param['batch_size'], self.param['beam_size']))
        for i in xrange(self.param['batch_size']):
            max_reward = -100.0
            max_index = -1
            for j in xrange(self.param['beam_size']):
                if rewards[i][j] > max_reward:
                    max_reward = rewards[i][j]
                    max_index = j
            if max_index != -1 and max_reward > 0:
                ml_rewards[i][max_index] = 1.0
            if max_index != -1 and max_reward < 0:
                ml_rewards[i][max_index] = -1.0
        return ml_rewards

    def get_data_and_feed_dict(self, batch_dict, epoch, overall_step_count):
        all_predicted, batch_ids, batch_orig_context, batch_context_nonkb_words, batch_context_kb_words, \
        batch_context_entities, batch_context_types, batch_context_rel, batch_context_ints, \
        batch_orig_response, batch_response_entities, batch_response_ints, batch_response_bools, batch_response_type, batch_required_argtypes, \
        variable_mask, variable_embed, variable_atten, kb_attention, variable_value_table = self.read_data.get_batch_data(batch_dict)

        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['epoch_for_feasible_program_at_last_step'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['epoch_for_feasible_program_at_last_step'][1]):
            feasible_program_at_last_step = 1.
            print 'Using feasible_program_at_last_step'
        else:
            feasible_program_at_last_step = 0.
        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['epoch_for_biasing_program_sample_with_target'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['epoch_for_biasing_program_sample_with_target'][1]):
            print 'Using program biasing with target'
            bias_prog_sampling_with_target = 1.
        else:
            bias_prog_sampling_with_target = 0.
        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['epoch_for_biasing_program_sample_with_last_variable'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['epoch_for_biasing_program_sample_with_last_variable'][1]):
            bias_prog_sampling_with_last_variable = 1.
        else:
            bias_prog_sampling_with_last_variable = 0.
        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['relaxed_reward_till_epoch'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['relaxed_reward_till_epoch'][1]):
            relaxed_reward_multipler = 0.
        else:
            if self.param['params_turn_on_after']=='epoch':
                relaxed_reward_multipler = (self.param['relaxed_reward_till_epoch'][0]-epoch)/float(self.param['relaxed_reward_till_epoch'][0])
                relaxed_reward_multipler = np.clip(relaxed_reward_multipler, 0, 1)
            elif self.param['params_turn_on_after']=='batch':
                relaxed_reward_multipler = (self.param['relaxed_reward_till_epoch'][1]-overall_step_count)/float(self.param['relaxed_reward_till_epoch'][1])
                relaxed_reward_multipler = np.clip(relaxed_reward_multipler, 0, 1)
        epsilon = 0
        if self.param['params_turn_on_after']=='epoch' and self.param['explore'][0] > 0:
            epsilon = self.param["initial_epsilon"]*np.clip(1.0-(epoch/self.param['explore'][0]),0,1)
        elif self.param['params_turn_on_after']=='batch' and self.param['explore'][1] > 0:
            epsilon = self.param["initial_epsilon"]*np.clip(1.0-(overall_step_count/self.param['explore'][1]),0,1)
        PruneNow = 0
        if (self.param['params_turn_on_after']=='epoch' and epoch >= self.param['prune_after_epoch_no.'][0]) or (self.param['params_turn_on_after']=='batch' and overall_step_count >= self.param['prune_after_epoch_no.'][1]):
            PruneNow = 1
        feed_dict1 = self.feeding_dict1(batch_context_nonkb_words, batch_context_kb_words, variable_mask, \
                                variable_embed, variable_atten, kb_attention, batch_response_type, batch_required_argtypes,\
                                 feasible_program_at_last_step, bias_prog_sampling_with_target, bias_prog_sampling_with_last_variable,\
                                  relaxed_reward_multipler, epsilon, PruneNow)
        feed_dict1['batch_ids'] = batch_ids
        feed_dict1['all_predicted'] = all_predicted
        return feed_dict1, batch_orig_context, batch_response_entities, batch_response_ints, batch_response_bools, variable_value_table

    def perform_test(self, batch_dict, epoch, overall_step_count):
        with torch.no_grad():
            feed_dict1, batch_orig_context, batch_response_entities, batch_response_ints, batch_response_bools, variable_value_table = self.get_data_and_feed_dict(batch_dict, epoch, overall_step_count)
            #a_seq, program_probabilities, per_step_probs = self.sess.run([self.action_sequence, self.program_probs, self.per_step_probs], feed_dict=feed_dict1)
            a_seq, program_probabilities, logprogram_probabilities, beam_props, per_step_probs, entropy = self.model(feed_dict1)
            # reshaping per_step_probs for printability
            per_step_probs = per_step_probs.data.cpu().numpy()
            [reward, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
             mask_intermediate_rewards] = self.forward_pass_interpreter(
                                             feed_dict1['preprocessed_var_mask_table'], feed_dict1['all_predicted'], feed_dict1['batch_ids'], batch_orig_context, a_seq, per_step_probs, \
                                             program_probabilities, variable_value_table, batch_response_entities, \
                                             batch_response_ints, batch_response_bools, epoch, overall_step_count)
            reward = np.array(reward)
            if overall_step_count%10==0:
                json.dump(self.ert_train_data, open(self.param['model_dir']+'/ert_train_data.json','w'), indent=1)
		print 'Dumped ert data in ', self.param['model_dir']+'/ert_train_data.json'
            relaxed_rewards = np.array(relaxed_rewards)
            reward[reward<0.] = 0.
            self.print_reward(reward)
            if overall_step_count%10==0:
                pkl.dump(self.evaluation_data, open(self.param['model_dir']+'/results.pkl','w'))
		print 'Dumped evaluation data in ', self.param['model_dir']+'/results.pkl'
            return sum(reward[:,0])/float(self.param['batch_size']), sum(np.max(reward,axis=1))/float(self.param['batch_size']), 0


      

    def print_reward(self, reward):
        batch_size = len(reward)
        beam_size= len(reward[0])
        best_reward_till_beam = {i:0.0 for i in xrange(beam_size)}
        avg_reward_at_beam = {i:0.0 for i in xrange(beam_size)}
        for batch_id in xrange(batch_size):
            for beam_id in xrange(beam_size):
                best_reward_till_beam[beam_id] += float(max(reward[batch_id][:(beam_id+1)]))
                avg_reward_at_beam[beam_id] += float(reward[batch_id][beam_id])
        best_reward_till_beam = {k:v/float(batch_size) for k,v in best_reward_till_beam.items()}
        avg_reward_at_beam = {k:v/float(batch_size) for k,v in avg_reward_at_beam.items()}
        for k in xrange(beam_size):
            print 'for beam ', k, ' best reward till this beam', best_reward_till_beam[k], ' (avg reward at this beam =', avg_reward_at_beam[k], ')'

    def remove_bad_data(self, data):
        for index, d in enumerate(data[:]):
           utter = d[0].lower()
           utter_yes_no_removed = utter.replace('yes','').replace('no','')
           utter_yes_no_removed = re.sub(' +',' ',utter_yes_no_removed)
           utter_yes_no_removed = utter_yes_no_removed.translate(string.maketrans("",""), string.punctuation).strip()
           if 'no, i meant' in utter or 'could you tell me the answer for that?' in utter or len(utter_yes_no_removed)<=1:
               data.remove(d)
	   elif ('gen_set' in self.questype_program_dict[self.param['question_type']] and sum(d[12])==0) or ('gen_map1' in self.questype_program_dict[self.param['question_type']] and sum(d[13])==0):
	       if self.param['question_type'].strip().lower() in d[11].lower():
	               data_id = '.'.join(d[15].split('.')[:3])
        	       self.evaluation_data[data_id] = [[0.0,0.0,False]]
               data.remove(d)
           #elif ('gen_set' in self.questype_program_dict[self.param['question_type']] and sum(d[12])==0) or ('gen_map1' in self.questype_program_dict[self.param['question_type']] and sum(d[13])==0):
           #    data.remove(d)
        return data
    
    def add_data_id(self, data):
        if len(data[0])>=16:
            return data
        for i in xrange(len(data)):
            data[i].append('')
        return data
    
    def get_batch_size_per_type(self, data_map):
        num_data_types = len(data_map)
        batch_size_types = {qtype:int(float(self.param['batch_size'])/float(num_data_types)) for qtype in data_map}
        diff = self.param['batch_size'] - sum(batch_size_types.values())
        qtypes = data_map.keys()
        count = 0
        while diff>0 and count<len(qtypes):
            batch_size_types[qtypes[count]]+=1
            count += 1
            if count == len(qtypes):
                count = 0
            diff -= 1
        if sum(batch_size_types.values())!=self.param['batch_size']:
            raise Exception("sum(batch_size_types.values())!=self.param['batch_size']")
        return batch_size_types

    def get_batch(self, i, data, data_map, batch_size_types):
        if not self.qtype_wise_batching:
            batch_dict = data[i*self.param['batch_size']:(i+1)*self.param['batch_size']]
            if len(batch_dict)<self.param['batch_size']:
                batch_dict.extend(data[:self.param['batch_size']-len(batch_dict)])
        else:
            batch_dict = []
            for qtype in data_map:
                data_map_qtype = data_map[qtype][i*batch_size_types[qtype]:(i+1)*batch_size_types[qtype]]
                if len(data_map_qtype)<batch_size_types[qtype]:
                    data_map_qtype.extend(data_map[qtype][:batch_size_types[qtype]-len(data_map_qtype)])
                batch_dict.extend(data_map_qtype)
            if len(batch_dict)!=self.param['batch_size']:
                raise Exception("len(batch_dict)!=self.param['batch_size']")
        return batch_dict

    def train(self):
        start = time.time()
        overall_step_count = self.starting_overall_step_count
        epoch = self.starting_epoch
        avg_valid_reward_at0, avg_valid_reward = self.perform_full_test(epoch, overall_step_count)
        print 'Epoch ', epoch, ' Validation over... overall avg. valid reward (over all)', avg_valid_reward, ' valid reward (at top beam)', avg_valid_reward_at0
        pkl.dump(self.evaluation_data, open(self.param['model_dir']+'/results.pkl','w'))
        print 'Dumped evaluation data in ', self.param['model_dir']+'/results.pkl'
	json.dump(self.ert_train_data, open(self.param['model_dir']+'/ert_train_data.json','w'), indent=1)
        print 'Dumped ert data in ', self.param['model_dir']+'/ert_train_data.json'
        end = time.time()
        print 100*'--'
        print end-start
        print 100*'--'


def main():
    torch.backends.cudnn.benchmark = True
    params_file = sys.argv[1]
    param = json.load(open(params_file))
    train_model = LoadModel(param)
    train_model.train()

    #else:
    #    print 'no GPU available'

if __name__=="__main__":
    freeze_support()
    main()




