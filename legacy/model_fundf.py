from __future__ import division, print_function

__author__ = "Lauri Juvela, lauri.juvela@aalto.fi", "Manu Airaksinen, manu.airaksinen@aalto.fi"

import math
import numpy as np
import tensorflow as tf

_FLOATX = tf.float32

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    if shape is None:
        return tf.get_variable(name)
    else:  
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)): 
    if shape is None:
        return tf.get_variable(name) 
    else:     
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)
   

class CNET():

    def __init__(self,
                 name,
                 residual_channels=128,
                 filter_width=5,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=512,
                 output_channels=301,
                 postnet_channels=256):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels
            
        self._name = name
        self._create_variables()

    def _create_variables(self):

        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels

        with tf.variable_scope(self._name):

            with tf.variable_scope('input_layer'):
                get_weight_variable('W', (1, self.input_channels, r)) # Input_channels = waveform -> fully connected matrix to learn a linear transformation
                get_bias_variable('b', (r))     

            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('conv_modules'):
                    with tf.variable_scope('module{}'.format(i)):
                        # (filter_width x input_channels x output_channels) 
                        get_weight_variable('filter_gate_W', (fw, r, 2*r)) 
                        get_bias_variable('filter_gate_b', (2*r)) 
                        
                        get_weight_variable('skip_weight_W', (1, r, r))
                        get_weight_variable('skip_weight_b', (r))

                        get_weight_variable('output_weight_W', (1, r, r))
                        get_weight_variable('output_weight_b', (r))

                                            
            with tf.variable_scope('postproc_module'):
                # (filter_width x input_channels x output_channels) 
                
                get_weight_variable('W1', (fw, r, s)) 
                get_bias_variable('b1', s)

                get_weight_variable('W2', (fw, s, self.output_channels)) 
                get_bias_variable('b2', self.output_channels)

    def get_variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)          


    def _input_layer(self, main_input, training=False):
        with tf.variable_scope('input_layer'):

            W = get_weight_variable('W')
            b = get_bias_variable('b')

            X = main_input
            X = tf.layers.dropout(inputs=X, rate=0.4, training=training)
            Y = tf.nn.convolution(X, W, padding='SAME')
            Y += b
            Y = tf.tanh(Y)

        return Y

    def _conv_module(self, main_input, residual_input, module_idx, dilation):
        with tf.variable_scope('conv_modules'):
            with tf.variable_scope('module{}'.format(module_idx)):

                W = get_weight_variable('filter_gate_W') 
                b = get_bias_variable('filter_gate_b') 
                r = self.residual_channels

                W_skip =  get_weight_variable('skip_weight_W')
                b_skip =  get_weight_variable('skip_weight_b')
 
                W_out = get_weight_variable('output_weight_W')
                b_out =  get_weight_variable('output_weight_b')

                X = main_input

                # convolution
                Y = tf.nn.convolution(X, W, padding='SAME', dilation_rate=[dilation])
                Y += b

                # filter and gate
                Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])
                
                # add residual channel
                skip_out = tf.nn.convolution(Y, W_skip, padding='SAME') # 1x1 convolution
                skip_out += b_skip

                Y = tf.nn.convolution(Y, W_out, padding='SAME')
                Y += b_out
                Y += X

        return Y, skip_out

    def _postproc_module(self, residual_module_outputs):
        with tf.variable_scope('postproc_module'):

            W1 = get_weight_variable('W1')
            b1 = get_bias_variable('b1')
            W2 = get_weight_variable('W2')
            b2 = get_bias_variable('b2')

            # sum of residual module outputs
            X = tf.zeros_like(residual_module_outputs[0])
            for R in residual_module_outputs:
                X += R

            Y = tf.nn.convolution(X, W1, padding='SAME')    
            Y += b1
            Y = tf.nn.relu(Y)

            Y = tf.nn.convolution(Y, W2, padding='SAME')    
            Y += b2
            
        return Y


    def forward_pass(self, X_input, training=False):
        skip_outputs = []
        with tf.variable_scope(self._name, reuse=True):
            R = self._input_layer(X_input, training=training)
            X = R
            for i, dilation in enumerate(self.dilations):
                X, skip = self._conv_module(X, R, i, dilation)
                skip_outputs.append(skip)

            Y = self._postproc_module(skip_outputs)
            Y = tf.reshape(Y,[-1,self.output_channels])

        return Y












