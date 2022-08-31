import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np
from data_prep import wrapperGenerator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['font.family'] = [u'serif']
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = 14, 10



class GAN():
    def __init__(self, train_events, train_norm1, train_norm2, 
                 noise_dim=50, 
                 events_shape=2, 
                 params_shape=6, 
                 learning_rate=1e-5, 
                 batch_size=10000, 
                 model_load_Path=None, 
                 start_epoch=0, 
                 truth=None, 
                 parmin=None, 
                 parmax=None,
                output_directory=""):
        
        self.noise_dim = noise_dim
        self.events_shape = events_shape
        self.params_shape = params_shape
        self.train_events = train_events
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate/10)
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.truth = truth
        self.train_norms1 = np.repeat(train_norm1, self.batch_size)
        self.train_norms2 = np.repeat(train_norm2, self.batch_size)
        self.parmin = parmin.astype(np.float32) 
        self.parmax = parmax.astype(np.float32)
        self.output_dir = output_directory
        
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        if model_load_Path != None:
            print("Loading Models...")
            self.discriminator.load_weights(os.path.join(model_load_Path, "2022.08.21_discriminator_6Params.h5"))
            self.generator.load_weights(os.path.join(model_load_Path, "2022.08.21_generator_6Params.h5"))
            print("Trained weights loaded...")
                
        self.discriminator.compile(loss='mse',
            optimizer=self.d_optimizer,
            metrics=['accuracy'])

        # The generator takes noise as input and generated the events 
        z = Input(shape=(self.noise_dim,))
        e, n1, n2 = self.generator(z)
        print(e.shape)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The valid takes generated images as input and determines validity
        valid = self.discriminator(e)
        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(z, [valid, n1, n2])
        self.combined.compile(loss='mse', optimizer=self.g_optimizer)#, loss_weights=[1, 1, 1])

    def build_generator(self):
        "Build the generator model"
        inputs = Input(shape=(self.noise_dim,))
        hidden = Dense(128)(inputs)
        hidden = LeakyReLU(alpha=0.2)(hidden)

        hidden = Dense(128)(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)

        hidden = Dense(128)(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)

        hidden = Dense(128)(hidden)
        hidden = LeakyReLU(alpha=0.2)(hidden)

        output = Dense(self.params_shape, activation="sigmoid")(hidden)

        output = Lambda(lambda x: wrapperGenerator(x, self.parmin, self.parmax, 1))(output)
        generator = Model(inputs=inputs, outputs=output)
        generator.summary()
        events = generator(inputs)
        
        return Model(inputs, events)


    def build_discriminator(self):
        "build the discriminator model"
        
        events = Input(shape=(self.events_shape,))
        h = Dense(256, input_shape=(self.events_shape,))(events)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dropout(0.05)(h)
        h = Dense(256)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dropout(0.05)(h)
        h = Dense(256)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dropout(0.05)(h)
        h = Dense(128)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dropout(0.05)(h)
        h = Dense(32)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(1, activation="sigmoid")(h)
        
        model = Model(events, h)
        model.summary()
        return model

    def train(self, epochs, sample_interval=50):

        fake = np.zeros((self.batch_size, 1)) 
        valid = tf.ones((self.batch_size, 1))
        
        for epoch in range(self.start_epoch, epochs):
            
            indices = np.random.randint(0, self.train_events.shape[1], self.batch_size)
            evnts = np.transpose(self.train_events[:, indices])

            noise = tf.random.normal((self.batch_size, self.noise_dim), 0.5, 0.3)

            events, norm1, norm2 = self.generator.predict_on_batch(noise)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(evnts, valid)
            d_loss_fake = self.discriminator.train_on_batch(events, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch(noise, [valid, self.train_norms1, self.train_norms2])
                
                
            if epoch % 10 == 0:
                print("At epoch ", epoch, " Norms", tf.reduce_mean(norm1), tf.reduce_mean(norm2), "g_loss: ", g_loss[0], " d_loss: ", d_loss[0])

            if epoch % sample_interval == 0:
                
                print("Inside sample interval block...")
                self.generator.save_weights(os.path.join(self.output_dir, "2022.08.21_generator_6Params.h5"))
                self.discriminator.save_weights(os.path.join(self.output_dir, "2022.08.21_discriminator_6Params.h5"))
                print("models saved...")
                
                num = 100000
                gen_events, norm1, norm2 = self.generate(num)
                print("Events generated...")
                print("training norms", self.train_norms1[0], self.train_norms2[0])
                plt.clf()
                fig = plt.figure(figsize=(14,6))
                ax1 = fig.add_subplot(1,2,1)
                ax1.hist(gen_events[:, 0], bins=100, alpha=0.5, label="sigma1: GAN", density=True)
                ax1.hist(self.train_events[0][:num], bins=100, alpha=0.5, label="sigma1: Sim", density=True)
                ax1.legend()
                ax2 = fig.add_subplot(1,2,2)
                ax2.hist(gen_events[:, 1], bins=100, alpha=0.5, label="sigma2: GAN", density=True)
                ax2.hist(self.train_events[1][:num], bins=100, alpha=0.5, label="sigma2: Sim", density=True)
                ax2.legend()
                plt.title("At Epoch - "+str(epoch))
                epoch_str = str(epoch).zfill(6)
                plt.savefig(os.path.join(self.output_dir, "Events_"+epoch_str+".png"))
                plt.show()
                
                get_1st_layer_output = K.function([self.generator.layers[0].input],
                                  [self.generator.layers[1].layers[9].output])
        
                layer_output = []
                
                noise = tf.random.normal((100000, self.noise_dim), 0.5, 0.2)
                p = get_1st_layer_output(noise)
                layer_output = p[0]

                results = layer_output
                print(results.shape)
                plt.clf()
                plt.figure(figsize=(20,15))
                for i in range(results.shape[1]):
                    plt.subplot(results.shape[1], 1, i+1)
                    plt.hist(results[:, i], bins=100, range=(0, 1), histtype='step', color = 'red')
                    truth_index = i
                    plt.axvline(x = self.truth[truth_index])
                    plt.title('param '+str(truth_index))
                plt.savefig(os.path.join(self.output_dir, "Params_"+epoch_str+".png"))
                plt.show()

    
    def generate(self, n=None):
        if n == None:
            n = self.train_events.shape[1]
        print("Generating: ", n, " events...")
        input_noise = tf.random.normal((n, self.noise_dim), 0.5, 0.2)
        events, norm1, norm2 = self.generator(input_noise)
        print("Generated Norms: ", norm1[:10], norm2[:10])
        return events.numpy(), norm1.numpy(), norm2.numpy()