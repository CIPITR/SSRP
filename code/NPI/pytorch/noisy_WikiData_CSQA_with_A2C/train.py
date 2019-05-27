from model import NPI
from read_data import ReadBatchData
from model import *
from interpreter import Interpreter
import numpy as np
import json
import random
import sys
import cPickle as pkl
import os, glob
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import math

class TrainModel():
    def __init__(self, param):
        self.param = param
        self.read_data = ReadBatchData(param)
        print "initialized read data"
        self.interpreter = Interpreter(self.read_data.program_type_vocab, self.read_data.argument_type_vocab)
        print "initialized interpreter"
        self.train_data = []
        if not isinstance(param['train_data_file'], list):
            self.training_files = [param['train_data_file']]
        else:
            self.training_files = param['train_data_file']
            random.shuffle(self.training_files)
        print 'Training data loaded'
        sys.stdout.flush()
        self.valid_data = []
        if not isinstance(param['valid_data_file'], list):
            self.valid_files = [param['valid_data_file']]
        else:
            self.valid_files = param['valid_data_file']
        for file in self.valid_files:
            self.valid_data.extend(pkl.load(open(file)))
        if not os.path.exists(param['model_dir']):
            os.mkdir(param['model_dir'])
        self.model_file = os.path.join(param['model_dir'],"best_model")
        with tf.Graph().as_default():
            self.model = NPI(param, self.read_data.none_argtype_index, self.read_data.num_argtypes, self.read_data.num_progs, self.read_data.max_arguments, self.read_data.rel_index, self.read_data.type_index, self.read_data.wikidata_rel_embed, self.read_data.wikidata_type_embed, self.read_data.vocab_init_embed, self.read_data.program_to_argtype, self.read_data.program_to_targettype)
            self.model.create_placeholder()
            self.action_sequence, self.program_probs, self.gradients = self.model.reinforce()
            self.train_op = self.model.train()
            print 'model created'
            sys.stdout.flush()
            self.saver = tf.train.Saver()
            init = tf.initialize_all_variables()
            self.sess = tf.Session()#tf_debug.LocalCLIDebugWrapperSession(tf.Session())
            if len(glob.glob(os.path.join(param['model_dir'], '*')))>0:
                print "best model exists .. restoring from there "
                self.saver.restore(self.sess, self.model_file)
            else:
                print "initializing fresh variables"
                self.sess.run(init)


    def feeding_dict1(self, encoder_inputs_w2v, encoder_inputs_kb_emb, variable_mask, variable_embed, kb_attention):
        feed_dict = {}
        for model_enc_inputs_w2v, enc_inputs_w2v in zip(self.model.encoder_text_inputs_w2v, encoder_inputs_w2v):
            feed_dict[model_enc_inputs_w2v] = enc_inputs_w2v
        feed_dict[self.model.encoder_text_inputs_kb_emb] = encoder_inputs_kb_emb
        print 'variable_mask', variable_mask.shape
        for i in range(variable_mask.shape[0]):
            for j in range(variable_mask.shape[1]):
                feed_dict[self.model.preprocessed_var_mask_table[i][j]] = variable_mask[i][j]
        for i in range(variable_embed.shape[0]):
            for j in range(variable_embed.shape[1]):
                feed_dict[self.model.preprocessed_var_emb_table[i][j]] = variable_embed[i][j]
        feed_dict[self.model.kb_attention] = kb_attention
#        for i in range(len(self.model.parameters)):
#            feed_dict[self.model.grad_values[i]] = np.zeros(self.model.grad_values[i].get_shape(), dtype=np.float32)
        return feed_dict

    def feeding_dict2(self, grad_values):
        feed_dict = {}
        assert len(self.model.grad_values) == len(grad_values)
        for model_grad_val_i, grad_val_i in zip(self.model.grad_values, grad_values):
            feed_dict[model_grad_val_i] = grad_val_i
        return feed_dict

    def map_multiply(self, arg):
        orig_shape = arg[0].shape
        arg0 = np.reshape(arg[0], (self.param['batch_size']*self.param['beam_size'], -1))
        arg1 = np.reshape(arg[1], (self.param['batch_size']*self.param['beam_size'],1))
        mul = np.reshape(np.multiply(arg0, arg1), orig_shape)
        return np.sum(mul,axis=(0,1))

    def generate_threads_for_interpreter(self,a_seq, variable_value_table):
        print len(a_seq['argument_table_index']), len(a_seq['argument_table_index'][0]), a_seq['argument_table_index'][0][0]
        keys = ['program_type','argument_type','target_type','target_table_index','argument_table_index']
        batch_length_set_sequences = []
        for i in range(self.param['batch_size']):
            new_dict = dict.fromkeys(keys)
            new_dict['program_type'] = [a_seq['program_type'][j][i] for j in range(len(a_seq['program_type']))]
            new_dict['argument_type'] = [a_seq['argument_type'][j][i] for j in range(len(a_seq['argument_type']))]
            new_dict['target_type'] = [a_seq['target_type'][j][i] for j in range(len(a_seq['target_type']))]
            new_dict['target_table_index'] = [a_seq['target_table_index'][j][i] for j in range(len(a_seq['target_table_index']))]
            new_dict['argument_table_index'] = [a_seq['argument_table_index'][j][i] for j in range(len(a_seq['argument_table_index']))]
            new_dict['variable_value_table'] = variable_value_table[i]
            batch_length_set_sequences.append(new_dict)
        return batch_length_set_sequences

    def perform_training(self, batch_dict):
        batch_orig_context, batch_context_nonkb_words, batch_context_kb_words, \
        batch_context_entities, batch_context_types, batch_context_rel, batch_context_ints, \
        batch_orig_response, batch_response_entities, batch_response_ints, batch_response_bools, \
        variable_mask, variable_embed, variable_atten, kb_attention, variable_value_table = self.read_data.get_batch_data(batch_dict)

        feed_dict1 = self.feeding_dict1(batch_context_nonkb_words, batch_context_kb_words, variable_mask, variable_embed, kb_attention)
        a_seq, program_probabilities, grad = self.sess.run([self.action_sequence, self.program_probs, self.gradients], feed_dict=feed_dict1)
#        print 100*'$'
#        print 'program_probabilities shape:',program_probabilities.shape
#        print 'grad shape:',grad.values()[0].shape
        data_for_interpreter = self.generate_threads_for_interpreter(a_seq, variable_value_table)
        target_value, target_type_id, Flag = zip(*map(self.interpreter.execute_multiline_program,data_for_interpreter))
        reward = self.interpreter.calculate_reward(target_value, target_type_id, Flag, batch_response_entities, batch_response_ints, batch_response_bools)
        reward = np.reshape(reward,[self.param['batch_size'], self.param['beam_size']])
        current_baseline = np.sum(np.multiply(program_probabilities,reward), axis=1, keepdims = True)
        rescaling_term_grad = np.subtract(np.array(reward),current_baseline)
#        print 'reward shape:',reward.shape
#        print'current_baseline shape:',current_baseline.shape
#        print'rescaling_term_grad shape:',rescaling_term_grad.shape
#        print 100*'$'

#        grad_values = map(self.map_multiply,zip(grad.values(),rescaling_term_grad))
        grad_values = [self.map_multiply([grad[x],rescaling_term_grad]) for x in grad.keys()]
        feed_dict2 = self.feeding_dict2(grad_values)

        self.sess.run([self.train_op], feed_dict=feed_dict2)
        return reward

    def train(self):
        best_valid_loss = float("inf")
        best_valid_epoch = 0
        last_overall_avg_train_loss = None
        overall_step_count = 0
        for epoch in range(self.param['max_epochs']):
            len_train_data = 0.
            train_loss = 0.
            for file in self.training_files:
                train_data = pkl.load(open(file))
                len_train_data = len_train_data + len(train_data)
                random.shuffle(train_data)
                n_batches = int(math.ceil(float(len(train_data))/float(self.param['batch_size'])))
                print 'number of batches ', n_batches, 'len train data ', len(train_data), 'batch size' , self.param['batch_size']
                sys.stdout.flush()
                for i in range(n_batches):
                    overall_step_count = overall_step_count + 1
                    train_batch_dict = train_data[i*self.param['batch_size']:(i+1)*self.param['batch_size']]
                    if len(train_batch_dict)<self.param['batch_size']:
                        train_batch_dict.extend(train_data[:self.param['batch_size']-len(train_batch_dict)])
                    sum_batch_loss = sum(self.perform_training(train_batch_dict))
                    avg_batch_loss = sum_batch_loss / float(self.param['batch_size'])
                    if overall_step_count%self.param['print_train_freq']==0:
                        print ('Epoch  %d Step %d train loss (avg over batch) =%.6f' %(epoch, i, avg_batch_loss))
                        sys.stdout.flush()
                        train_loss = train_loss + sum_batch_loss
                        avg_train_loss = float(train_loss)/float(i+1)
                    overall_step_count += 1
            overall_avg_train_loss = train_loss/float(len_train_data)
            print 'epoch ',epoch,' of training is completed ... overall avg. train loss ', overall_avg_train_loss

def main():
    param = json.load(open('parameters.json'))
    print 'loaded params '
    train_model = TrainModel(param)
    train_model.train()

if __name__=="__main__":
    main()