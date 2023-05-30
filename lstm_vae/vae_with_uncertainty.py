import numpy as np

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, Concatenate
from keras.layers.core import Dense, Lambda
from keras import metrics as objectives


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
            `x_decoded_logit_sigma`: examples x timesteps x features: logit(sigma) of above; this is the "uncertainty" in the reconstruction
            `combined_decoder`: examples x timesteps x features*2: concatenation of the above two tensors

    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,), name="input")

    # LSTM encoding
    h = LSTM(intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_sigma_factor = Dense(latent_dim)(h)
    # [-1, 1] not [0, 1] but the way we use it, the sign is irrelevant
    
    def sampling(args):
        z_mean, z_sigma_factor = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_sigma_factor * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_logit_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_sigma_factor])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True, name='decoder_mean')
    decoder_logit_sigma = LSTM(input_dim, return_sequences=True, name='decoder_logit_sigma')
    #use "logit" transform to map [-1, 1] to [-inf, inf].

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    x_decoded_logit_sigma = decoder_logit_sigma(h_decoded)
    combined_x_decoded = Concatenate(axis=2, name="combined_decoder")([x_decoded_mean, x_decoded_logit_sigma])
    print(type(combined_x_decoded))
    
    # end-to-end autoencoder
    vae = Model(x, [x_decoded_mean, x_decoded_logit_sigma, combined_x_decoded])

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    def combined_loss(x_possibly_backwards, outputs):
        x_decoded_mean, x_decoded_logit_sigma = outputs[:, :, :input_dim], outputs[:, :, input_dim:]
        x_decoded_logit_sigma = x_decoded_logit_sigma * .4999 + .5 # map [-1, 1] to [0, 1]
        x_decoded_sigma = K.log(x_decoded_logit_sigma/(1-x_decoded_logit_sigma))
        xent_loss = K.mean(K.square((x_possibly_backwards - x_decoded_mean) * x_decoded_sigma)) * .5 + K.mean(x_decoded_logit_sigma / (1 - x_decoded_logit_sigma))
        kl_loss = - 0.5 * K.mean(1 - K.abs(z_sigma_factor) - K.square(z_mean) + K.log(K.abs(z_sigma_factor) + K.epsilon()))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss={"combined_decoder": combined_loss})
    
    return vae, encoder, generator

