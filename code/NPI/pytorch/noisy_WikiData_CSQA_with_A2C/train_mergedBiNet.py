#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:04:19 2018

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""

from model_mergedBiNet import NPI
from read_data import ReadBatchData
import itertools
from interpreter import Interpreter
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



class TrainModel():
    def __init__(self, param):
        np.random.seed(1)
        torch.manual_seed(999)
        #if torch.cuda.is_available(): torch.cuda.manual_seed_all(999)
        self.param = param
        self.run_interpreter = True
        self.run_validation = False
        self.generate_data = False
        self.param = param
        if not self.generate_data and os.path.exists(self.param['model_dir']+'/model_data.pkl'):
            self.pickled_train_data = pkl.load(open(self.param['model_dir']+'/model_data.pkl'))
        else:
            self.pickled_train_data = {}
        self.starting_epoch = 0
        self.starting_overall_step_count = 0
        self.starting_validation_reward_overall = 0
        self.starting_validation_reward_topbeam = 0
        if 'caching' not in self.param:
            self.param['caching'] = False
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
        if 'none_decay' not in self.param:
            self.param['none_decay'] = 0
        if 'update_past_model_freq' not in self.param:
            self.param['update_past_model_freq'] = 5
        if 'dump_ert_train_data_freq' not in self.param:
            self.param['dump_ert_train_data_freq'] = 50
        if 'train_mode' not in self.param:
            self.param['train_mode'] = 'reinforce'
        self.qtype_wise_batching = self.param['questype_wise_batching']
        if 'use_variable_permutation' not in self.param:
            self.param['use_variable_permutation'] = False
        if 'use_backprop_annealing' not in self.param:
            self.param['use_backprop_annealing'] = False
        if 'backprop_annealing_upto_batch' not in self.param or self.param['backprop_annealing_upto_batch'] == 0:
            self.param['use_backprop_annealing'] = False
        else:
            self.param['use_backprop_annealing'] = True
        if 'use_question_template' not in self.param or not self.param['use_question_template']:
            self.param['use_question_template'] = False

        else:
            if 'use_kb_emb_in_ques_rnn' not in self.param:
                self.param['use_kb_emb_in_ques_rnn'] = False
            if self.param['use_kb_emb_in_ques_rnn']:
                raise Exception('cannot use kb embedding in question rnn when using question template')
            if not os.path.exists(self.param['template_vocab_file']):
                raise Exception('cannot run on question template if template_vocab.pkl file is missing')
        self.read_data = ReadBatchData(self.param)
        if not self.param['use_question_template']:
            vocab_init_embed = self.read_data.vocab_init_embed
        else:
            vocab_init_embed = self.read_data.template_init_embed
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
        if 'reward_function' not in self.param:
            reward_func = "jaccard"
            self.param['reward_function'] = "jaccard"
        else:
            reward_func = self.param['reward_function']
        if 'relaxed_reward_strict' not in self.param:
            relaxed_reward_strict = False
            self.param['relaxed_reward_strict'] = relaxed_reward_strict
        else:
            relaxed_reward_strict = self.param['relaxed_reward_strict']
        if self.param['parallel']==1:
            raise Exception('Need to fix the intermediate rewards for parallelly executing interpreter')
        for k,v in self.param.items():
            print 'PARAM: ', k , ':: ', v
        print 'loaded params '
        self.train_data = []
        if os.path.isdir(self.param['train_data_file']):
            self.training_files = [self.param['train_data_file']+'/'+x for x in os.listdir(self.param['train_data_file']) if x.endswith('.pkl')]
        elif not isinstance(self.param['train_data_file'], list):
            self.training_files = [self.param['train_data_file']]
        else:
            self.training_files = self.param['train_data_file']
            random.shuffle(self.training_files)
        self.valid_data = []
        if os.path.isdir(param['valid_data_file']):
            self.valid_files = [self.param['valid_data_file']+'/'+x for x in os.listdir(self.param['valid_data_file']) if x.endswith('.pkl')]
        elif not isinstance(self.param['valid_data_file'], list):
            self.valid_files = [self.param['valid_data_file']]
        else:
            self.valid_files = self.param['valid_data_file']
        for file in self.valid_files:
            d = pkl.load(open(file))
            file_basename = os.path.basename(file)
            d = self.add_data_id(d, file_basename)
            self.valid_data.extend(d)
        if self.qtype_wise_batching:
            self.valid_data_map = self.read_data.get_data_per_questype(self.valid_data)
            self.valid_batch_size_types = self.get_batch_size_per_type(self.valid_data_map)
            self.n_valid_batches = int(math.ceil(float(sum([len(x) for x in self.valid_data_map.values()])))/float(self.param['batch_size']))
        else:
            self.n_valid_batches = int(math.ceil(float(len(self.valid_data))/float(self.param['batch_size'])))

        if not os.path.exists(self.param['model_dir']):
            os.mkdir(self.param['model_dir'])
        learning_rate = self.param['learning_rate']
        start = time.time()
        if torch.cuda.is_available():
            USE_DEVICE = 'cuda'
        else:
            USE_DEVICE = 'cpu'
        self.model = NPI(self.param, self.read_data.none_argtype_index, self.read_data.num_argtypes, \
                         self.read_data.num_progs, self.read_data.max_arguments, \
                         self.read_data.rel_index, self.read_data.type_index, \
                         self.read_data.wikidata_rel_embed, self.read_data.wikidata_type_embed, \
                         vocab_init_embed, self.read_data.program_to_argtype, \
                         self.read_data.program_to_targettype, USE_DEVICE)
        self.past_model_data = {}
        self.checkpoint_prefix = os.path.join(param['model_dir'], param['model_file'])
        if os.path.exists(self.checkpoint_prefix):
            self.model.load_state_dict(torch.load(self.checkpoint_prefix))
            fr = open(self.param['model_dir']+'/metadata.txt').readlines()
            self.starting_epoch = int(fr[0].split(' ')[1].strip())
            self.starting_overall_step_count = int(fr[1].split(' ')[1].strip())
            self.starting_validation_reward_overall = float(fr[2].split(' ')[1].strip())
            self.starting_validation_reward_topbeam = float(fr[3].split(' ')[1].strip())
            print 'restored model'
        self.ert_train_data = {}
        end = time.time()
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=[0.9,0.999], weight_decay=1e-5)
        print self.model

        self.past_model = NPI(self.param, self.read_data.none_argtype_index, self.read_data.num_argtypes, \
                         self.read_data.num_progs, self.read_data.max_arguments, \
                         self.read_data.rel_index, self.read_data.type_index, \
                         self.read_data.wikidata_rel_embed, self.read_data.wikidata_type_embed, \
                         vocab_init_embed, self.read_data.program_to_argtype, \
                         self.read_data.program_to_targettype,'cpu').to('cpu')

        print 'model created in ', (end-start), 'seconds'

        self.interpreter = Interpreter(self.param['wikidata_dir'], self.param['num_timesteps'], \
                                       self.read_data.program_type_vocab, self.read_data.argument_type_vocab, self.printing, terminate_prog, relaxed_reward_strict, reward_function = reward_func, boolean_reward_multiplier = boolean_reward_multiplier, relaxed_reward_till_epoch=relaxed_reward_till_epoch, unused_var_penalize_after_epoch=unused_var_penalize_after_epoch)
        if self.param['parallel'] == 1:
            self.InterpreterProxy, self.InterpreterProxyListener = proxy.createProxy(self.interpreter)
            self.interpreter.parallel = 1
            self.lock = Lock()
        print "initialized interpreter"

    def perform_full_validation(self, epoch, overall_step_count):
        valid_reward = 0
        valid_reward_at0 = 0
        for i in xrange(self.n_valid_batches):
            train_batch_dict = self.get_batch(i, self.valid_data, self.valid_data_map, self.valid_batch_size_types)
            avg_batch_reward_at0, avg_batch_reward, _ = self.perform_validation(train_batch_dict, epoch, overall_step_count)
            if i%self.param['print_valid_freq']==0 and i>0:
                valid_reward += avg_batch_reward
                valid_reward_at0 += avg_batch_reward_at0
                avg_valid_reward = float(valid_reward)/float(i+1)
                avg_valid_reward_at0 = float(valid_reward_at0)/float(i+1)
                print ('Valid Results in Epoch  %d Step %d (avg over batch) valid reward (over all) =%.6f valid reward (at top beam)=%.6f running avg valid reward (over all)=%.6f running avg valid reward (at top beam)=%.6f' %(epoch, i, avg_batch_reward, avg_batch_reward_at0, avg_valid_reward, avg_valid_reward_at0))
        overall_avg_valid_reward = valid_reward/float(self.n_valid_batches)
        overall_avg_valid_reward_at0 = valid_reward_at0/float(self.n_valid_batches)
        return overall_avg_valid_reward_at0, overall_avg_valid_reward

    def feeding_dict2(self, reward, IfPosIntermediateReward, mask_IntermediateReward, IntermediateReward, relaxed_rewards, mask, instance_weight):
        feed_dict = {}
        feed_dict['reward'] = reward
        feed_dict['IfPosIntermediateReward'] = IfPosIntermediateReward
        feed_dict['mask_IntermediateReward'] = mask_IntermediateReward
        feed_dict['IntermediateReward'] = IntermediateReward
        feed_dict['Relaxed_reward'] = relaxed_rewards
        feed_dict['mask_reinforce'] = mask
        feed_dict['instance_weight'] = instance_weight
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
        feed_dict['ert_subgraph'] = [feed_dict['kb_attention'][i][self.read_data.program_type_vocab['gen_set']].reshape((self.read_data.max_num_var, self.read_data.max_num_var, self.read_data.max_num_var)) for i in range(kb_attention.shape[0])]
        feed_dict['trt_subgraph'] = [feed_dict['kb_attention'][i][self.read_data.program_type_vocab['gen_map1']].reshape((self.read_data.max_num_var, self.read_data.max_num_var, self.read_data.max_num_var)) for i in range(kb_attention.shape[0])]
        feed_dict['ert_subgraph'] = np.asarray(feed_dict['ert_subgraph'])
        feed_dict['trt_subgraph'] = np.asarray(feed_dict['trt_subgraph'])
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
        #print 'entities prepopulated ', [variable_value_table[self.read_data.ent_index][i] for i in xrange(self.param['max_num_var']) if prepopulated_var_mask[self.read_data.ent_index][i]==1]
        #print 'relations prepopulated ', [variable_value_table[self.read_data.rel_index][i] for i in xrange(self.param['max_num_var']) if prepopulated_var_mask[self.read_data.rel_index][i]==1]
        #print 'types prepopulated ', [variable_value_table[self.read_data.type_index][i] for i in xrange(self.param['max_num_var']) if prepopulated_var_mask[self.read_data.type_index][i]==1]

        for i in xrange(self.param['num_timesteps']):
            if a_seq['program_type'][i]==self.read_data.program_type_vocab['gen_set'] or a_seq['program_type'][i]==self.read_data.program_type_vocab['gen_map1']:
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

    def forward_pass_interpreter(self, ert_subgraph, trt_subgraph, preprocessed_var_mask, batch_ids, batch_orig_context, a_seq, per_step_probs, \
                                 program_probabilities, variable_value_table, batch_response_entities, \
                                 batch_response_ints, batch_response_bools, epoch_number, overall_step_count):
        program_probabilities = program_probabilities.data.cpu().numpy()
        preprocessed_var_mask = np.asarray(preprocessed_var_mask).transpose((2,0,1))
        #Reward_From_Model = np.transpose(np.array(a_seq['Model_Reward_Flag']))

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
                new_a_seq[batch_id][beam_id][key] = np.where(preprocessed_var_mask[batch_id]==1, variable_value_table[batch_id], None)


        [handle_variable_value_table(key,beam_id,timestep,batch_id) for (key,beam_id,timestep,batch_id) in list(itertools.product\
         (keys,xrange(self.param['beam_size']),xrange(self.param['num_timesteps']),xrange(self.param['batch_size'])))]

        for batch_id in xrange(self.param['batch_size']):
            for beam_id in xrange(self.param['beam_size']):
                new_a_seq[batch_id][beam_id]['program_probability'] = program_probabilities[batch_id][beam_id]


        rewards = []
        rewards_to_print = []
        intermediate_rewards_flag = []
        mask_intermediate_rewards = []
        intermediate_rewards = []
        relaxed_rewards = []
        for batch_id in xrange(self.param['batch_size']):
            if self.printing:
                print 'batch id ', batch_id, ':: Query :: ', batch_orig_context[batch_id]
            data_id = batch_ids[batch_id]
            rewards_batch = []
            rewards_batch_to_print = []
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
                best_reward = -1000
                best_max_intermediate_reward = -1000
                best_relaxed_reward = -1000
                best_intermediate_mask = -1000
                best_intermediate_reward_flag = -1000
                best_to_print = ''
                args = (new_a_seq[batch_id][beam_id], \
                           batch_response_entities[batch_id], \
                           batch_response_ints[batch_id], \
                           batch_response_bools[batch_id])
                reward_to_print, max_intermediate_reward, relaxed_reward, intermediate_mask, intermediate_reward_flag, to_print = self.interpreter.calculate_program_reward(args, epoch_number, overall_step_count)
                ents, rels, types = self.get_ert_from_a_seq(new_a_seq[batch_id][beam_id], variable_value_table[batch_id], preprocessed_var_mask[batch_id])
                best_reward = reward_to_print
                best_max_intermediate_reward = max_intermediate_reward
                best_relaxed_reward = relaxed_reward
                best_intermediate_mask = intermediate_mask
                best_intermediate_reward_flag = intermediate_reward_flag
                best_to_print = to_print
                ents_best = ents
                rels_best = rels
                types_best = types
                if reward_to_print>=0 and reward_to_print<=0.3:
                    if self.param['use_variable_permutation']:
                        new_argval_table = self.get_all_variable_permutations(batch_id, beam_id, new_a_seq, preprocessed_var_mask, ert_subgraph, trt_subgraph)
                    else:
                        new_argval_table = None
                    if new_argval_table is not None:
                        print 'trying ',new_argval_table.shape[0], 'permutations ...',
                        for k in xrange(new_argval_table.shape[0]):
                            new_a_seq[batch_id][beam_id]['argument_table_index'] = new_argval_table[k]
                            args = (new_a_seq[batch_id][beam_id], \
                                   batch_response_entities[batch_id], \
                                   batch_response_ints[batch_id], \
                                   batch_response_bools[batch_id])
                            reward, max_intermediate_reward, relaxed_reward, intermediate_mask, intermediate_reward_flag, to_print = self.interpreter.calculate_program_reward(args, epoch_number, overall_step_count)
                            ents, rels, types = self.get_ert_from_a_seq(new_a_seq[batch_id][beam_id], variable_value_table[batch_id], preprocessed_var_mask[batch_id])
                            if reward > best_reward or max_intermediate_reward > best_max_intermediate_reward:
                                best_reward = reward
                                best_max_intermediate_reward = max_intermediate_reward
                                best_relaxed_reward = relaxed_reward
                                best_intermediate_mask = intermediate_mask
                                best_intermediate_reward_flag = intermediate_reward_flag
                                best_to_print = to_print
                                ents_best = ents
                                rels_best = rels
                                types_best = types
                            if best_reward >= 0.3 and k>3:
                                print 'Out of that, did only ', k,
                                break
                        if best_reward > reward_to_print:
                            print ' .. increased reward from ',reward_to_print , ' to ',best_reward,
                        print ''
                if self.printing:
                    print best_to_print
                if best_reward>0:
                    if data_id not in self.ert_train_data:
                        self.ert_train_data[data_id] = {}
                    elif 'reward' not in self.ert_train_data[data_id] or best_reward > self.ert_train_data[data_id]['reward']:
                        d = {'reward':best_reward, 'entities':list(ents_best), 'relations':list(rels_best), 'types':list(types_best)} #add batch id here
                        self.ert_train_data[data_id] = d
                rewards_batch.append(best_reward)
                rewards_batch_to_print.append(reward_to_print)
                intermediate_rewards_flag_batch.append(best_intermediate_reward_flag)
                relaxed_rewards_batch.append(best_relaxed_reward)
                mask_intermediate_rewards_batch.append(best_intermediate_mask)
                intermediate_rewards_batch.append(best_max_intermediate_reward)
                #print 'per_step_programs', [self.read_data.program_type_vocab_inv[x] for x in new_a_seq[batch_id][beam_id]['program_type']]

            rewards.append(rewards_batch)
            rewards_to_print.append(rewards_batch_to_print)
            intermediate_rewards_flag.append(intermediate_rewards_flag_batch)
            mask_intermediate_rewards.append(mask_intermediate_rewards_batch)
            intermediate_rewards.append(intermediate_rewards_batch)
            relaxed_rewards.append(relaxed_rewards_batch)
        rewards = np.array(rewards)
        rewards_to_print = np.array(rewards_to_print)
        #if self.param['reward_from_model']:
        #    rewards = np.where(Reward_From_Model == 0, rewards, -1*np.ones_like(rewards))
        intermediate_rewards = np.array(intermediate_rewards)
        intermediate_rewards_flag = np.array(intermediate_rewards_flag)
        mask_intermediate_rewards = np.array(mask_intermediate_rewards)
        relaxed_rewards = np.array(relaxed_rewards)
        return rewards_to_print, rewards, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, mask_intermediate_rewards



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
        batch_ids, batch_context_template_str, batch_orig_context, batch_context_template, batch_context_nonkb_words, batch_context_kb_words, \
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
        feed_dict1['batch_template_string'] = batch_context_template_str
        return feed_dict1, batch_orig_context, batch_response_entities, batch_response_ints, batch_response_bools, variable_value_table

    def get_all_variable_permutations(self, batch_id, beam_id, a_seq, preprocessed_var_mask, ert_subgraph, trt_subgraph):
        i = batch_id
        j = beam_id
        l_e = []
        l_r = []
        l_t = []
        for m in xrange(self.param['num_timesteps']):
            l_e.extend([a_seq[i][j]['argument_table_index'][m][k] for k in xrange(self.read_data.max_arguments) if a_seq[i][j]['argument_type'][m][k]==self.read_data.ent_index and preprocessed_var_mask[i][self.read_data.ent_index][a_seq[i][j]['argument_table_index'][m][k]]==1])
            l_r.extend([a_seq[i][j]['argument_table_index'][m][k] for k in xrange(self.read_data.max_arguments) if a_seq[i][j]['argument_type'][m][k]==self.read_data.rel_index and preprocessed_var_mask[i][self.read_data.rel_index][a_seq[i][j]['argument_table_index'][m][k]]==1])
            l_t.extend([a_seq[i][j]['argument_table_index'][m][k] for k in xrange(self.read_data.max_arguments) if a_seq[i][j]['argument_type'][m][k]==self.read_data.type_index and preprocessed_var_mask[i][self.read_data.type_index][a_seq[i][j]['argument_table_index'][m][k]]==1])
            if a_seq[i][j]['program_type'][m]==self.read_data.program_type_vocab['gen_set']:
                e = a_seq[i][j]['argument_table_index'][m][0]
                r = a_seq[i][j]['argument_table_index'][m][1]
                t = a_seq[i][j]['argument_table_index'][m][2]
                if ert_subgraph[i][e][r][t]==0:
                    return None
            elif a_seq[i][j]['program_type'][m]==self.read_data.program_type_vocab['gen_map1']:
                t1 = a_seq[i][j]['argument_table_index'][m][0]
                r = a_seq[i][j]['argument_table_index'][m][1]
                t2 = a_seq[i][j]['argument_table_index'][m][2]
                if trt_subgraph[i][t1][r][t2]==0:
                    return None
        l_e = list(set(l_e))
        l_r = list(set(l_r))
        l_t = list(set(l_t))
        if len(l_e)>0:
            other_perm_e = self.find_other_permutations(len(l_e))
        else:
            other_perm_e = [[0]]
            l_e = [0]
        if len(l_r)>0:
            other_perm_r = self.find_other_permutations(len(l_r))
        else:
            other_perm_r = [[0]]
            l_r = [0]
        if len(l_t)>0:
            other_perm_t = self.find_other_permutations(len(l_t))
        else:
            other_perm_t = [[0]]
            l_t = [0]
        #print 'l_e',l_e, 'perm_e_i', other_perm_e
        #print 'l_r', l_r, 'perm_r_i', other_perm_r
        #print 'l_t', l_t, 'perm_t_i', other_perm_t
        new_argval_table = []
        for perm_e_i in other_perm_e:
                e_dict = {}
                for l_e_i, perm_e_ij in zip(l_e, perm_e_i):
                    e_dict[l_e_i] = perm_e_ij
                for perm_r_i in other_perm_r:
                    r_dict = {}
                    for l_r_i, perm_r_ij in zip(l_r, perm_r_i):
                        r_dict[l_r_i] = perm_r_ij
                    for perm_t_i in other_perm_t:
                        t_dict = {}
                        for l_t_i, perm_t_ij in zip(l_t, perm_t_i):
                            t_dict[l_t_i] = perm_t_ij
                            new_argval_table_i = np.zeros((self.param['num_timesteps'], self.read_data.max_arguments), dtype=np.int32)
                            ditch_perm = False
                            if ditch_perm:
                                break
                            for m in xrange(self.param['num_timesteps']):
                                if ditch_perm:
                                    break
                                for k in xrange(self.read_data.max_arguments):
                                    if a_seq[i][j]['argument_type'][m][k]==self.read_data.ent_index and a_seq[i][j]['argument_table_index'][m][k] in e_dict:
                                        new_argval_table_i[m][k] = e_dict[a_seq[i][j]['argument_table_index'][m][k]]
                                    elif a_seq[i][j]['argument_type'][m][k]==self.read_data.rel_index and a_seq[i][j]['argument_table_index'][m][k] in r_dict:
                                        new_argval_table_i[m][k] = r_dict[a_seq[i][j]['argument_table_index'][m][k]]
                                    elif a_seq[i][j]['argument_type'][m][k]==self.read_data.type_index and a_seq[i][j]['argument_table_index'][m][k] in t_dict:
                                        new_argval_table_i[m][k] = t_dict[a_seq[i][j]['argument_table_index'][m][k]]
                                    else:
                                        new_argval_table_i[m][k] = a_seq[i][j]['argument_table_index'][m][k]
                                if a_seq[i][j]['program_type'][m]==self.read_data.program_type_vocab['gen_set']:
                                    e = new_argval_table_i[m][0]
                                    r = new_argval_table_i[m][1]
                                    t = new_argval_table_i[m][2]
                                    if ert_subgraph[i][e][r][t]==0:
                                        ditch_perm = True
                                        break
                                elif a_seq[i][j]['program_type'][m]==self.read_data.program_type_vocab['gen_map1']:
                                    t1 = new_argval_table_i[m][0]
                                    r = new_argval_table_i[m][1]
                                    t2 = new_argval_table_i[m][2]
                                    if trt_subgraph[i][t1][r][t2]==0:
                                        ditch_perm = True
                                        break
                            if not ditch_perm:
                                new_argval_table.append(new_argval_table_i)
        if len(new_argval_table)>0:
            new_argval_table = np.asarray(new_argval_table)
            return new_argval_table
        else:
            return None
    def find_other_permutations(self, arr_len):
        possibilities = xrange(self.param['max_num_var'])
        permutations = itertools.permutations(possibilities, arr_len)
        return list(permutations)

    def perform_validation(self, batch_dict, epoch, overall_step_count):
        with torch.no_grad():
            feed_dict1, batch_orig_context, batch_response_entities, batch_response_ints, batch_response_bools, variable_value_table = self.get_data_and_feed_dict(batch_dict, epoch, overall_step_count)

            a_seq, program_probabilities, logprogram_probabilities, beam_props, per_step_probs, entropy = self.model(feed_dict1)
            # reshaping per_step_probs for printability
            per_step_probs = per_step_probs.data.cpu().numpy()

            [reward_print, reward, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
             mask_intermediate_rewards] = self.forward_pass_interpreter(feed_dict1['ert_subgraph'], feed_dict1['trt_subgraph'], \
                                             feed_dict1['preprocessed_var_mask_table'], feed_dict1['batch_ids'], feed_dict1['batch_template_string'], batch_orig_context, a_seq, per_step_probs, \
                                             program_probabilities, variable_value_table, batch_response_entities, \
                                             batch_response_ints, batch_response_bools, epoch, overall_step_count)
            reward = np.array(reward)
            relaxed_rewards = np.array(relaxed_rewards)
            reward_print = np.array(reward_print)
            reward_print[reward_print<0.] = 0.
            self.print_reward(reward_print)
            return sum(reward_print[:,0])/float(self.param['batch_size']), sum(np.max(reward_print,axis=1))/float(self.param['batch_size']), 0


        #def apply_gradients(self, model, optimizer, gradients):
        #    optimizer.apply_gradients(zip(gradients, self.model.variables))

    def perform_training(self, batch_dict, epoch, overall_step_count):
        print 'in perform_training'
        feed_dict1, batch_orig_context, batch_response_entities, batch_response_ints, batch_response_bools, variable_value_table = self.get_data_and_feed_dict(batch_dict, epoch, overall_step_count)
        self.optimizer.zero_grad()
        def dummy(input_model,feed_dict1):
            a_seq, program_probabilities, logprogram_probabilities, beam_props, per_step_probs, entropy = input_model(feed_dict1)
            # reshaping per_step_probs for printability
            per_step_probs = per_step_probs.data.cpu().numpy()

            [reward_print, reward, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
             mask_intermediate_rewards] =  self.forward_pass_interpreter(feed_dict1['ert_subgraph'], \
             feed_dict1['trt_subgraph'], feed_dict1['preprocessed_var_mask_table'], \
             feed_dict1['batch_ids'], batch_orig_context, a_seq, per_step_probs, program_probabilities, variable_value_table, \
             batch_response_entities, batch_response_ints, batch_response_bools, epoch, overall_step_count)

            reward = np.array(reward)
            relaxed_rewards = np.array(relaxed_rewards)
            instances_pos_rewards = set(list(np.where(reward!=0)[0]))
            if self.param['use_backprop_annealing']:
                instance_weights = np.asarray([1. if i in instances_pos_rewards else \
                                               min(1.,float(overall_step_count/float(1.+self.param['backprop_annealing_upto_batch']))) \
                                               for i in xrange(reward.shape[0])])
            else:
                instance_weights = np.ones(reward.shape[0], dtype=np.float32)
            program_type = a_seq['program_type']
            return reward, reward_print, program_probabilities, logprogram_probabilities, per_step_probs, entropy, \
                    intermediate_rewards_flag, mask_intermediate_rewards, \
                    intermediate_rewards, relaxed_rewards, instance_weights,program_type

        [reward_orig, reward_print, program_probabilities, logprogram_probabilities, \
         per_step_probs, entropy, intermediate_rewards_flag, mask_intermediate_rewards, \
         intermediate_rewards, relaxed_rewards,instance_weights,Cur_P_type_orig] = dummy(self.model,feed_dict1)
        self.blockPrint()

        if self.param['caching'] and all([id_i in self.past_model_data and any([x[2]<overall_step_count-self.param['update_past_model_freq'] for x in self.past_model_data[id_i]]) for id_i in feed_dict1['batch_template_string']]):
            reference_r = []
            Ref_P_type = []
            for id_i in feed_dict1['batch_template_string']:
                ref_r = None
                ref_p = None
                for past_data_i in self.past_model_data[id_i].reverse():
                    if past_data_i[2]<overall_step_count-self.param['update_past_model_freq']:
                        ref_r = past_data_i[0]
                        ref_p = past_data_i[1]
                        break
                reference_r.append(ref_r)
                Ref_P_type.append(ref_p)
            #Ref_P_type is a batch_size dim list of beam_size dim list of num_timesteps dim tensor
            Ref_P_type = map(list, zip(*Ref_P_type))
            #Ref_P_type is a beam_size dim list of batch_size dim list of num_timesteps dim tensor
            Ref_P_type = [torch.unbind(torch.stack(prog_type_i, dim=1),dim=0) for prog_type_i in Ref_P_type]
            print 'after retrieving ', len(Ref_P_type), len(Ref_P_type[0])
            reference_r = np.array(reference_r)
        else:
            temp_out = dummy(self.past_model,feed_dict1)
            reference_r = temp_out[0]
            Ref_P_type = temp_out[-1]
        self.enablePrint()
        final_1 =[]
        final_2 =[]
        for beam_id in range(self.param['beam_size']):
            final_1.append(torch.stack(Cur_P_type_orig[beam_id],dim=0))
            final_2.append(torch.stack(Ref_P_type[beam_id],dim=0))

        Cur_P_type = torch.stack(final_1,dim=0).permute(1,2,0).type(torch.float32)
        Ref_P_type = torch.stack(final_2,dim=0).to('cuda').permute(1,2,0).type(torch.float32)

        Cur_P_type = torch.stack(self.param['beam_size']*[Cur_P_type],dim = 3)
        Ref_P_type = torch.stack(self.param['beam_size']*[Ref_P_type],dim = 2)

        reward = torch.tensor(reward_orig,dtype=torch.float32).to('cuda')
        reference_r = torch.tensor(reference_r,dtype=torch.float32).to('cuda')
        r = torch.stack(self.param['beam_size']*[reward],dim= 2)
        reference_r = torch.stack(self.param['beam_size']*[reference_r],dim = 1)
        temp = r-reference_r+1e-2
        mask = torch.where(temp>0,temp,torch.zeros_like(r))
        distance_mask = torch.where(Cur_P_type!=Ref_P_type,torch.ones_like(Cur_P_type),torch.zeros_like(Cur_P_type))
        distance_mask = torch.sum(distance_mask ,dim = 0)
        # distance_mask has shape batch_size x beam_size x beam_size

        distance_mask = torch.exp(-3*distance_mask.type(torch.float32).cpu())
        mask = mask.cpu()*distance_mask

        mask = torch.max(mask,dim = 2)[0]
        feed_dict2 = self.feeding_dict2(reward, intermediate_rewards_flag, mask_intermediate_rewards, intermediate_rewards, \
                                        relaxed_rewards, mask, instance_weights)
        loss = -self.model.train(feed_dict2)
        self.optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 50)


        # Calling the step function on an Optimizer makes an update to its
        # parameters
        self.optimizer.step()

        if self.param['caching']:
            Cur_P_type_orig = [torch.unbind(torch.stack(prog_type_i, dim=1), dim=0) for prog_type_i in Cur_P_type_orig]
            #Cur_P_type_orig is a beam_size dim list of batch_size dim list of num_timesteps dim tensor
            Cur_P_type_orig = map(list, zip(*Cur_P_type_orig))
            print 'when saving ',len(Cur_P_type_orig), len(Cur_P_type_orig[0])
            for i,id_i in enumerate(feed_dict1['batch_template_string']):
                if id_i not in self.past_model_data:
                    self.past_model_data[id_i].append([reward_orig[i], Cur_P_type_orig[i], overall_step_count])

        if overall_step_count%self.param['dump_ert_train_data_freq']==0:
            json.dump(self.ert_train_data, open(self.param['model_dir']+'/ert_train_data.json','w'), indent=1)
        reward_print = np.array(reward_print)
        reward_print[reward_print<0.] = 0.
        self.print_reward(reward_print)
        output_loss = loss.detach().cpu().numpy()
        del loss,  feed_dict2, feed_dict1, intermediate_rewards, relaxed_rewards, intermediate_rewards_flag, \
                 mask_intermediate_rewards, program_probabilities, logprogram_probabilities, per_step_probs, entropy
        return sum(reward_print[:,0])/float(self.param['batch_size']), sum(np.max(reward_print,axis=1))/float(self.param['batch_size']),  output_loss/float(self.param['batch_size'])




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
            elif self.param['question_type']!='verify' and sum(d[12])==0 and sum(d[13])==0:
                data.remove(d)
        return data

    def add_data_id(self, data, filename):
        if len(data[0])==16:
            return data
        for i in xrange(len(data)):
            data[i].append(filename+'_'+str(i))
        if not len(data[0])==16:
            raise Exception('Something wrong ')
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
        print 'batch_size_types' ,batch_size_types
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
        last_overall_avg_train_loss = 0

        last_avg_valid_reward = self.starting_validation_reward_overall
        last_avg_valid_reward_at0 = self.starting_validation_reward_topbeam
        overall_step_count = self.starting_overall_step_count
        epochwise_step_count = 0
        self.qtype_wise_batching = True

        for e in xrange(self.param['max_epochs']):
            start = time.time()
            epoch = self.starting_epoch+e
            epochwise_step_count =0
            #self.epsilon = self.param["initial_epsilon"]*(1.0-(epoch)/(self.param['max_epochs']))
            if self.param['train_mode']=='ml' and 'change_train_mode_after_epoch' in self.param and epoch >= self.param['change_train_mode_after_epoch']:
                self.param['train_mode']='reinforce'
            num_train_batches = 0.
            train_loss = 0.
            train_reward = 0.
            train_reward_at0 = 0.
            for file in self.training_files:
                #with tf.device('/cpu:0'):
                train_data = pkl.load(open(file))
                file_basename = os.path.basename(file)
                train_data = self.add_data_id(train_data, file_basename)
                train_data = self.remove_bad_data(train_data)
                random.shuffle(train_data)
                if self.qtype_wise_batching:
                        train_data_map = self.read_data.get_data_per_questype(train_data)
                        if len(train_data_map)==0:
                            continue
                        batch_size_types = self.get_batch_size_per_type(train_data_map)
                        n_batches = int(math.ceil(float(sum([len(x) for x in train_data_map.values()])))/float(self.param['batch_size']))
                else:
                        train_data_map = None
                        n_batches = int(math.ceil(float(len(train_data))/float(self.param['batch_size'])))
                print 'Number of batches ', n_batches, 'len train data ', len(train_data), 'batch size' , self.param['batch_size']
                for i in xrange(n_batches):
                    #with tf.device('/cpu:0'):
                    num_train_batches+=1.
                    train_batch_dict = self.get_batch(i, train_data, train_data_map, batch_size_types)
                    avg_batch_reward_at0, avg_batch_reward, sum_batch_loss = self.perform_training(train_batch_dict, epoch, overall_step_count)
                    #with tf.device('/cpu:0'):
                    avg_batch_loss = sum_batch_loss / float(self.param['batch_size'])
                    if overall_step_count%self.param['update_past_model_freq']==0:
                        cpu_model_dict = {}
                        model_dict = self.model.state_dict()
                        for key, val in model_dict.items():
                            cpu_model_dict[key] = val.cpu()
                        self.past_model.load_state_dict(cpu_model_dict)

                    if overall_step_count%self.param['print_train_freq']==0:
                            train_loss = train_loss + sum_batch_loss
                            train_reward += avg_batch_reward
                            train_reward_at0 += avg_batch_reward_at0
                            avg_train_reward = float(train_reward)/float(num_train_batches)
                            avg_train_reward_at0 = float(train_reward_at0)/float(num_train_batches)
                            print ('Epoch  %d Step %d (avg over batch) train loss =%.6f  train reward (over all) =%.6f train reward (at top beam)=%.6f running avg train reward (over all)=%.6f running avg train reward (at top beam)=%.6f' %(epoch, epochwise_step_count, avg_batch_loss, avg_batch_reward, avg_batch_reward_at0, avg_train_reward, avg_train_reward_at0))
                    if overall_step_count%self.param['valid_freq']==0 and overall_step_count>self.starting_overall_step_count:
                        print 'Going for validation'
                        avg_valid_reward_at0, avg_valid_reward = self.perform_full_validation(epoch, overall_step_count)
                        #with tf.device('/cpu:0'):
                        print 'Epoch ', epoch, ' Validation over... overall avg. valid reward (over all)', avg_valid_reward, ' valid reward (at top beam)', avg_valid_reward_at0
                        if avg_valid_reward_at0>last_avg_valid_reward_at0:
                                with open(self.param['model_dir']+'/metadata.txt','w') as fw:
                                    fw.write('Epoch_number '+str(epoch)+'\n')
                                    fw.write('overall_step_count '+str(overall_step_count)+'\n')
                                    fw.write('Avg_valid_reward '+str(avg_valid_reward)+'\n')
                                    fw.write('avg_valid_reward_at0 '+str(avg_valid_reward_at0)+'\n')
                                torch.save(self.model.state_dict(), self.checkpoint_prefix)
                                last_avg_valid_reward_at0 = avg_valid_reward_at0
                                print 'Saving Model in ', self.checkpoint_prefix
                    overall_step_count += 1
                    epochwise_step_count += 1
            #with tf.device('/cpu:0'):
            overall_avg_train_loss = train_loss/float(num_train_batches)
            if overall_avg_train_loss>last_overall_avg_train_loss:
                    print 'Avg train loss increased by ', (overall_avg_train_loss-last_overall_avg_train_loss), ' from ', last_overall_avg_train_loss, 'to', overall_avg_train_loss
            overall_avg_train_reward = train_reward/float(num_train_batches)
            overall_avg_train_reward_at0 = train_reward_at0/float(num_train_batches)
            print 'Epoch ',epoch,' of training is completed ... overall avg. train loss ', overall_avg_train_loss, ' train reward (over all)', \
            overall_avg_train_reward, ' train reward (top beam)', overall_avg_train_reward_at0
            end = time.time()
            print 100*'--'
            print end-start
            print 100*'--'


def main():
    #tf.enable_eager_execution(config=tf.ConfigProto(allow_soft_placement=True,
    #                                    log_device_placement=True), device_policy=tfe.DEVICE_PLACEMENT_SILENT)
    torch.backends.cudnn.benchmark = True
    params_file = sys.argv[1]
    timestamp = sys.argv[2]
    param = json.load(open(params_file))
    param['model_dir']= param['model_dir']+'/'+param['question_type']+'_'+timestamp
    train_model = TrainModel(param)
    train_model.train()

    #else:
    #    print 'no GPU available'

if __name__=="__main__":
    freeze_support()
    main()
