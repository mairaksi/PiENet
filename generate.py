from __future__ import division

__author__ = "Manu Airaksinen, manu.airaksinen@aalto.fi"


import os
import sys
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '' # Uncommment to force CPU
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wavfile

import model_fundf as model
import sp_module as sp



_FLOATX = tf.float32 

def generate(wav_list, target_dir, model_name):

    # Set input & output dimensions (DO NOT EDIT FOR TRAINED MODELS)
    winlen = 512
    hop = 160
    input_dim = winlen
    output_dim = 351
    
    # network config
    dilations=[1, 2, 4, 8, 1, 2, 4, 8]
    filter_width = 5
    residual_channels = 128
    postnet_channels = 256

    f0_model = model.CNET(name='f0model',  input_channels=input_dim,
                        output_channels=output_dim, dilations=dilations, filter_width=filter_width, 
                        residual_channels=residual_channels, postnet_channels=postnet_channels)   


    input_var = tf.placeholder(shape=(None, None, winlen), dtype=_FLOATX)

    #  model input is framed raw signal
    f0_activations = f0_model.forward_pass(input_var)
    f0_activations = tf.nn.softmax(f0_activations)
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        
        model_file = './saved_models/' + model_name + '_best.ckpt'
        print("Loading model: " + model_file)

        saver.restore(sess, model_file)
        print("Model restored.")

        for wfile in wav_list:
                fs, y = wavfile.read(wfile)
                y = np.float32(y/(2**15))

                if fs != 16000:
                    raise Exception('fs needs to be 16 kHz!')

                input_frames = np.reshape(sp.get_frames(y,winlen,hop),[1,-1,winlen])
 
                f0_act_np = sess.run([f0_activations], feed_dict={input_var: input_frames})
                f0_act_np = np.reshape(np.asarray(f0_act_np),[-1,output_dim])
                
                f0_gen_np = sp.getF0fromActivations(f0_act_np,minf0=50,maxf0=500,Nbins=351)

                basename = os.path.basename(os.path.splitext(wfile)[0])
                target_file = target_dir + '/' + basename + '.f0'
                #f0_gen_np.tofile(target_file) # Use this to write float32 binary files
                np.savetxt(target_file,f0_gen_np,fmt='%.2f') # Use this to write ASCII output files


if __name__=="__main__":

# Parse input arguments


    if(len(sys.argv) == 1): # No extra arguments, use default input and output files
        file_list =  'test_wavs.scp'
        target_dir = 'f0/'
    
    elif(len(sys.argv) == 2):  # Custom list of files for input
        target_dir = 'f0/'
        if(isinstance(sys.argv[1], str)):
            file_list = sys.argv[1]
        else:
            raise Exception('First input argument must be a string (file path)')
    elif(len(sys.argv) == 3):  # Custom list of files for input
        if(isinstance(sys.argv[1], str) and isinstance(sys.argv[2], str)):
            file_list = sys.argv[1]
            target_dir = sys.argv[2]
        else:
            raise Exception('First input argument must be a string (file path)')
    
    elif(len(sys.argv) > 3): # Custom list of input files + custom result file
            raise Exception('Too many input arguments')

    ########### GET DATA #############
    wasdir = 0
    # If input
    if(os.path.isdir(file_list)):
        fileList = sorted(glob.glob(file_list + '*.wav'))
        wasdir = 1
        if(len(fileList) == 0):
            raise Exception('Provided directory contains no .wav files')
    elif(file_list.endswith('.wav')):
        fileList = list()
        fileList.append(file_list)
    else:
        if(os.path.isfile(file_list)):
            #fileList = list(filter(bool,[line.rstrip('\n') for line in open(file_list)]))
            with open(file_list) as wavlist:
                fileList = wavlist.read().splitlines()
        else:
            raise Exception('Provided input file list does not exist.')


    # Model name (located in 'saved_models/' directory)
    model_name = 'gtaug'

    # Target directory for estimated F0 files


    os.makedirs(target_dir, exist_ok=True)

    generate(fileList, target_dir, model_name)
