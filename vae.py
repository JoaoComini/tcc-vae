from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class Vae:
    def __init__(self, sizes, loss=binary_crossentropy):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.input_dim = sizes[0]
        self.loss = loss
        
        inputs = Input(shape=(self.input_dim,), name='encoder_input')

        x = Dense(sizes[1], activation='relu')(inputs)

        for size in sizes[2:-1]:
            x = Dense(size, activation='relu')(x)

        z_mean = Dense(sizes[-1], name='z_mean')(x)
        z_log_var = Dense(sizes[-1], name='z_log_var')(x)
        z = Lambda(self.sampling, name='z')([z_mean, z_log_var])

        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        latent_inputs = Input(shape=(sizes[-1],), name='z_sampling')
        x = Dense(sizes[-2], activation='relu')(latent_inputs)

        for size in sizes[-3:0:-1]:
            x = Dense(size, activation='relu')(x)

        outputs = Dense(self.input_dim, name='decoder_output', activation='sigmoid')(x)

        self.decoder = Model(latent_inputs, outputs, name='decoder')

        outputs = self.decoder(self.encoder(inputs)[-1])
        self.model = Model(inputs, outputs, name='vae_mlp')

        self.model.compile(optimizer='adam', loss=self.vae_loss)

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def vae_loss(self, x, x_decoded):
        z_mean, z_log_var, _ = self.encoder.outputs

        reconstruction_loss = self.loss(x, x_decoded) * self.input_dim

        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss
