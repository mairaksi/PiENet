__author__ = "Manu Airaksinen, manu.airaksinen@aalto.fi"

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '' # Uncomment to force CPU

import numpy as np
import tensorflow as tf

import model_fundf as model
import soundfile
import augmentation
import sp_module as sp

_FLOATX = tf.float32 

def train_model(train_wav_list,train_f0_list,test_wav_list,test_f0_list,model_name='test'):
    tf.reset_default_graph() # debugging, clear all tf variables

    # Data dimensions
    winlen = 512 # samples
    hop = 160 # samples
    input_dim = winlen
    output_dim = 351
    f0_max = 500
    f0_min = 50

    downsample_f0 = False # If target f0 is computed with 5ms hop size (and network uses 10ms), set True

    # network config
    dilations=[1, 2, 4, 8, 1, 2, 4, 8]
    filter_width = 5
    residual_channels = 128
    postnet_channels = 256

    f0_model = model.CNET(name='f0model',  input_channels=input_dim,
                        output_channels=output_dim, dilations=dilations, filter_width=filter_width, 
                        residual_channels=residual_channels, postnet_channels=postnet_channels)   

    # data placeholders of shape (batch_size, timesteps, feature_dim)
    input_var = tf.placeholder(shape=(None, None, winlen), dtype=_FLOATX)
    output_var = tf.placeholder(shape=(None, None, output_dim), dtype=_FLOATX)
    training_var = tf.placeholder(dtype=tf.bool)

    # encoder model input is observed signal
    f0_activations = f0_model.forward_pass(input_var)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=output_var,logits=f0_activations))

    theta = f0_model.get_variable_list()
    optim = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.9,beta2=0.999).minimize(loss, var_list=[theta])
   
    # Add text file containing paths to augmentation noise samplesa
    # If wav_scp = None, uses white noise as additive noise augmentation
    noise_samples = augmentation.load_noise_samples(wav_scp=None)  

    with tf.Session() as sess:

        num_epochs = 100

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op) 

        saver = tf.train.Saver(max_to_keep=0)
        epoch_test_error_prev = 1e10

        epoch_loss = np.zeros((num_epochs,2),dtype=np.float32)

        for epoch in range(num_epochs):
            
            print("Training epoch {}".format(epoch))
            epoch_error = 0.0
            epoch_test_error = 0.0
            file_ind = 0
            wav_inds = np.random.permutation(len(train_wav_list))

            for i in wav_inds:
                wfile = train_wav_list[i]
                f0file = train_f0_list[i]

                y, _ = soundfile.read(wfile)
                y_noise = augmentation.add_noise_file(y,noise_samples)

                f0 = np.fromfile(f0file,dtype=np.float32)
                if downsample_f0:
                    f0 = f0[0::2] # Downsample F0 vector if computed with 5 ms hop size

                f0_onehot = sp.onehot(f0,minf0=f0_min,maxf0=f0_max,Nbins=output_dim) # Convert F0 vector to log-spaced onehot representation
                input_frames = np.reshape(sp.get_frames(y_noise,winlen,hop),[1,-1,winlen])
                _, loss_np = sess.run([optim, loss], feed_dict={input_var: input_frames, 
                                    output_var: f0_onehot, training_var: True})
                epoch_error += loss_np / np.float32(len(train_wav_list))
            
                file_ind += 1 
           
            print("Error for epoch %d: %f" % (epoch, epoch_error))
            saver.save(sess,"./saved_models/" + model_name + ".ckpt")

            # Validation set:

            for i in range(len(test_wav_list)):
                wfile = test_wav_list[i]
                f0file = test_f0_list[i]

                y, _ = soundfile.read(wfile)
                f0 = np.fromfile(f0file,dtype=np.float32)
                if downsample_f0:
                    f0 = f0[0::2]

                f0_onehot = sp.onehot(f0,minf0=f0_min,maxf0=f0_max,Nbins=output_dim)
                input_frames = np.reshape(sp.get_frames(y,winlen,hop),[1,-1,winlen])
 
                loss_np = sess.run([loss], feed_dict={input_var: input_frames, 
                                     output_var: f0_onehot, training_var: False})
                epoch_test_error += loss_np[0] / np.float32(len(test_wav_list))

            print("Test Error for epoch %d: %f" % (epoch, epoch_test_error))

            epoch_loss[epoch, 0] = epoch_error
            epoch_loss[epoch, 1] = epoch_test_error
            epoch_loss.tofile('./saved_models/' + model_name + '_loss.dat')

            if epoch_test_error < epoch_test_error_prev:
                saver.save(sess,"./saved_models/" + model_name + "_best.ckpt")
                epoch_test_error_prev = epoch_test_error
             
    

if __name__ == "__main__":
    wav_scp = 'wavs.scp'
    with open(wav_scp) as wavlist:
        wavs = wavlist.read().splitlines()

    f0_scp = 'f0s.scp'
    with open(f0_scp) as f0list:
        f0s = f0list.read().splitlines()
    
    # Split into train and test files
    Ntest = np.int32(np.round(0.1*len(wavs)))
    wav_inds = np.random.RandomState(seed=42).permutation(len(wavs))
    wavs = np.asarray(wavs)[wav_inds]
    f0s = np.asarray(f0s)[wav_inds]
    train_wavs = wavs[Ntest:]
    train_f0s = f0s[Ntest:]
    test_wavs = wavs[:Ntest]
    test_f0s = f0s[:Ntest]

    train_model(train_wavs, train_f0s, test_wavs, test_f0s, model_name='fundf-model')

  
