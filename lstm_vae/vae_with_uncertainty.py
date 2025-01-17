import numpy as np
import tensorflow as tf


import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, Concatenate, Layer
from keras.layers.core import Dense, Lambda
from keras import metrics as objectives
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.callbacks import Callback
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import MultiOptimizer

def page_row_to_vec(row):
    return [np.log(row.len) / 4,np.sqrt(row.left),row.offset,row.height,(row.left + row.width) ** 2,row.confidence / 100,row.near_bottom]

def pad_page_to(page, target=40):
    if len(page) < target:
        
        for i in range(target - len(page)):
            #pad at start, to better enable decoding in reverse order
            block = np.zeros(len(page[0]))
            block[-1] = -np.log(1 + i) #"here it comes" signal
            page = [block] + page #note that we're building backwards towards the start of the tensor, so in practice i is counting down
        return page
    if len(page) == target:
        return page
    first_half = page[:target//2]
    for i in range(len(first_half)):
        first_half[-i - 1][-1] = -np.log(1 + i)
    second_half = page[-target//2:]
    return first_half + second_half

PARTIALLY_UNMASK_FUDGE = 0.01
MIN_Z_SIGMA = 0.05
X_SIGMA_TRANSFORM_FACTOR = 0.49
EON_EPOCHS = 30
KL_START_EPOCH = 5
KL_ANNEAL_TIME = 9 #so 10 epochs of KL weight=min, 18 to anneal, ~30 epochs of weight=1, then repeat.
MIN_KL_WEIGHT = 0.1
BASE_LR = 0.002

def create_lstm_uvae(input_dim, 
    timesteps, 
    batch_size, 
    intermediate_dim, 
    latent_dim,
    epsilon_std=1.):

    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 

    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM. 
        latent_dim: int, latent z-layer shape. 
        epsilon_std: float, z-layer sigma.

    # Model
        inputs:
            `input`: examples x timesteps x features
            `input_backwards`: examples x timesteps x features (same as input, but *possibly* reversed)
        outputs:
            `x_decoded_mean`: examples x timesteps x features: reconstruction from input that should be the same as input_backwards
            `x_decoded_sigma_intervaltransform`: examples x timesteps x features: logit(sigma) of above; this is the "uncertainty" in the reconstruction
            `combined_decoder`: examples x timesteps x features*2: concatenation of the above two tensors

    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,), name="input")

    # LSTM encoding
    h = LSTM(intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim, name="z_mean")(h)
    z_sigma_factor = Dense(latent_dim, name="z_sigma_factor")(h)
    # [-1, 1] not [0, 1] but the way we use it, the sign is irrelevant
    
    weight = K.variable(0.0)

    def sampling(args):
        z_mean, z_sigma_factor = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        #print("sampling", z_mean.shape, z_sigma_factor.shape, epsilon.shape)
        return z_mean + (K.abs(z_sigma_factor) + MIN_Z_SIGMA) * epsilon #* weight  #don't temper by weight, because we're going to temper the lr instead

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_sigma_intervaltransform])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_sigma_factor])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True, name='decoder_mean')
    decoder_sigma_intervaltransform = LSTM(input_dim, return_sequences=True, name='decoder_sigma_intervaltransform')
    #use "interval transform" to map [-1, 1] to (0, inf).

    class RepeaterLayer(Layer):
        def __init__(self, repeat_count, **kwargs):
            self.repeat_count = repeat_count
            super(RepeaterLayer, self).__init__(**kwargs)
        def call(self, inputs):
            return RepeatVector(self.repeat_count)(inputs)
        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.repeat_count, input_shape[1])


    h_decoded = RepeaterLayer(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    x_decoded_sigma_intervaltransform = decoder_sigma_intervaltransform(h_decoded)
    combined_x_decoded = Concatenate(axis=2, name="combined_decoder")([x_decoded_mean, x_decoded_sigma_intervaltransform])
    print(type(combined_x_decoded))
    
    # end-to-end autoencoder
    vae = Model(x, [x_decoded_mean, x_decoded_sigma_intervaltransform, combined_x_decoded])

    for i, layer in enumerate(vae.layers):
        if layer.name == "z_sigma_factor":
            z_sig_layer_index = i
            break


    steady_optimizer = RMSprop(momentum = 0.05, learning_rate=BASE_LR)
    jerky_optimizer = RMSprop(learning_rate=BASE_LR)
    optimizers_and_layers = [(jerky_optimizer, vae.layers[z_sig_layer_index]), 
                             (steady_optimizer, vae.layers[:i] + vae.layers[i+1:])]
    optimizr = MultiOptimizer(optimizers_and_layers)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeaterLayer(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    class AnnealingCallback(Callback):
        def __init__(self, weight):
            self.weight = weight
        def on_epoch_end (self, epoch, logs={}):
            effective_epoch = epoch % EON_EPOCHS
            if effective_epoch == 0:
                K.set_value(self.weight, MIN_KL_WEIGHT)
            if (effective_epoch) >= KL_START_EPOCH :
                new_weight = min(K.get_value(self.weight) + ((1. - MIN_KL_WEIGHT)/ KL_ANNEAL_TIME), 1.)
                K.set_value(self.weight, new_weight)
            K.set_value(jerky_optimizer.learning_rate, BASE_LR * K.get_value(self.weight) * 1.4) #should average out to about BASE_LR
            print (" KL fac = " + str(K.get_value(self.weight)))


    
    def vae_loss(weight):
        def combined_loss(x_possibly_backwards, outputs):
            x_decoded_mean, x_decoded_sigma_intervaltransform = outputs[:, :, :input_dim], outputs[:, :, input_dim:]
            #print(x_possibly_backwards.shape, x_decoded_mean.shape, x_decoded_sigma_intervaltransform.shape)
            x_decoded_sigma_intervaltransform = x_decoded_sigma_intervaltransform * X_SIGMA_TRANSFORM_FACTOR + .5 # map [-1, 1] to [0, 1]
            x_decoded_sigma = (x_decoded_sigma_intervaltransform/(1-x_decoded_sigma_intervaltransform)) / 2 # map [0, 1] to [0, inf]
            mostly_mask =  (tf.sign(x_possibly_backwards) ** 2 + .01)
            #return K.sum(K.abs(outputs))
            xent_loss = (K.mean(K.square((x_possibly_backwards - x_decoded_mean) / x_decoded_sigma) * mostly_mask) * .5 + #only count loss on the "real" part of the input
                        K.mean(K.log(x_decoded_sigma) * mostly_mask) ) #add in the loss from the "uncertainty" in the reconstruction
            z_sigma_corrected = K.abs(z_sigma_factor) + MIN_Z_SIGMA
            kl_loss = - 0.5 * K.mean(1 - z_sigma_corrected - K.square(z_mean) + K.log(z_sigma_corrected))
            loss = xent_loss + weight * kl_loss
            return loss
        return combined_loss

    optimizr = RMSprop(momentum = 0.05)#learning_rate=1.0)
    vae.compile(optimizer=optimizr, loss={"combined_decoder": vae_loss(weight)})
    
    return vae, encoder, generator, AnnealingCallback(weight)

