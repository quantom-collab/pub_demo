import torch
import torch.nn as nn
import torch.optim as optim
from data_prep_torch import wrapperGenerator

class Generator(nn.Module):
    def __init__(self, input_size, ngpu=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(input_size, 128, bias=False),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128, bias=False),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128, bias=False),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128, bias=False),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 6, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, input_size, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(input_size, 128, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(128, 128, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(128, 128, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(128, 128, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(128, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    ''' Custom weights initialization called on Generator and Discriminator '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GAN():
    def __init__(self, train_events, train_norm1, train_norm2,
                 noise_mean=0.5,
                 noise_std=0.18,
                 params_dim=6,
                 events_dim=2,
                 learning_rate=1e-2,
                 batch_size=256,
                 truth=None,
                 parmin=None,
                 parmax=None,
                 output_dir=""):

        self.train_events = train_events
        self.train_norms1 = train_norm1.repeat(batch_size)
        self.train_norms2 = train_norm2.repeat(batch_size)

        self.noise_dim  = params_dim
        self.noise_mean = noise_mean
        self.noise_std  = noise_std
        self.params_dim = params_dim
        self.events_dim = events_dim
        assert self.events_dim == self.train_events.shape[1]

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.d_n_steps = 1
        self.g_n_steps = 1

        self.truth  = truth
        self.parmin = parmin
        self.parmax = parmax
        self.output_dir = output_dir

        # create generator and discriminator models
        self.generator     = Generator(self.params_dim)
        self.discriminator = Discriminator(self.events_dim)
        # initialize model weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        # print models
        print(self.generator)
        print(self.discriminator)

        # set beta1 hyperparameter for Adam optimizer
        opt_beta1 = 0.5
        opt_beta2 = 0.999
        # create optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(),     lr=learning_rate, betas=(opt_beta1, opt_beta2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(opt_beta1, opt_beta2))
        self.g_loss_function = nn.MSELoss()
        self.d_loss_function = nn.MSELoss()
        self.norm1_loss_function = nn.MSELoss()
        self.norm2_loss_function = nn.MSELoss()

    def train(self, n_epochs=10+1, sample_interval=10):
        label = torch.zeros((self.batch_size,))

        print("Starting Training Loop...")
        for epoch in range(n_epochs):
            indices = torch.randint(0, self.train_events.shape[0], (self.batch_size,))
            events_real = self.train_events[indices,:]
            noise = torch.normal(self.noise_mean, self.noise_std, size=(self.batch_size, self.noise_dim))

            # train discriminator
            for d_step in range(self.d_n_steps):
                self.d_optimizer.zero_grad()

                label.fill_(1.0)
                real_pred = self.discriminator(events_real).view(-1)
                d_loss_real = self.d_loss_function(label, real_pred)
                d_loss_real.backward()

                params = self.generator(noise)
                events_fake, norm1, norm2 = wrapperGenerator(params, self.parmin, self.parmax, 1)
                events_fake = torch.squeeze(events_fake)
                label.fill_(0.0)
                fake_pred = self.discriminator(events_fake).view(-1)
                d_loss_fake = self.d_loss_function(label, fake_pred)
                d_loss_fake.backward()

                d_loss = 0.5*(d_loss_real + d_loss_fake)
                self.d_optimizer.step()

            # train generator
            for g_step in range(self.g_n_steps):
                self.g_optimizer.zero_grad()
                params = self.generator(noise)
                events_fake, norm1, norm2 = wrapperGenerator(params, self.parmin, self.parmax, 1)
                events_fake = torch.squeeze(events_fake)
                label.fill_(1.0)
                fake_pred = self.discriminator(events_fake).view(-1)
                g_loss_fake = self.g_loss_function(label, fake_pred)

                g_loss = g_loss_fake
                g_loss.backward()
                self.g_optimizer.step()

            # print status

            if epoch % 100 == 0:
                print("Epoch:", epoch,
                      "; norms:", torch.mean(norm1).item(), torch.mean(norm2).item(),
                      "; g_loss:", g_loss.item(), " d_loss:", d_loss.item())
            if epoch % 1000 == 0:
                with torch.no_grad():
                    plot_events_and_params(epoch, self.generator,
                                           self.train_events, self.train_norms1, self.train_norms2,
                                           self.noise_mean, self.noise_std, self.noise_dim, self.batch_size,
                                           self.truth, self.parmin, self.parmax,
                                           self.output_dir)
                self.d_n_steps += 1
                self.d_n_steps = min(self.d_n_steps, 5)


def plot_events_and_params(epoch, generator,
                           train_events, train_norms1, train_norms2,
                           noise_mean, noise_std, noise_dim, batch_size,
                           truth, parmin, parmax,
                           output_dir):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    print("Inside sample interval block...")

    #generator.save_weights(os.path.join(output_dir, "2022.08.21_generator_6Params.h5"))
    #discriminator.save_weights(os.path.join(output_dir, "2022.08.21_discriminator_6Params.h5"))
    #print("models saved...")

    num = 10000
    gen_params = generate(num, generator, noise_mean, noise_std, noise_dim, batch_size)
    gen_events, norm1, norm2 = wrapperGenerator(gen_params, parmin, parmax, 1)
    gen_events = gen_events.numpy()
    print("Events generated...")
    print("training norms", train_norms1[0], train_norms2[0])
    plt.clf()
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,2,1)
    ax1.hist(gen_events[:, 0], bins=100, alpha=0.5, label="sigma1: GAN", density=True)
    ax1.hist(train_events[:num,0], bins=100, alpha=0.5, label="sigma1: Sim", density=True)
    ax1.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist(gen_events[:, 1], bins=100, alpha=0.5, label="sigma2: GAN", density=True)
    ax2.hist(train_events[:num,1], bins=100, alpha=0.5, label="sigma2: Sim", density=True)
    ax2.legend()
    plt.title("At Epoch - "+str(epoch))
    epoch_str = str(epoch).zfill(6)
    plt.savefig(os.path.join(output_dir, "Events_"+epoch_str+".png"))
    plt.show()

    results = gen_params.numpy()

    def plot_posterior_hist(generated_params):
        n,_,_ = plt.hist(generated_params, bins=200, range=(0, 1), histtype='step', density=True, color='red')
        return n

    def plot_normal_dist(mean, std, scale=None):
        if scale is None:
            scale = 1.0/(std*np.sqrt(2*np.pi))
        x = np.linspace(0, 1, 128)
        y = scale * np.exp(-(x - mean)**2/(2*std)**2)
        plt.plot(x, y, color='gray')

    plt.figure(figsize=(20,15))
    for i in range(results.shape[1]):
        plt.subplot(results.shape[1], 1, i+1)
        n = plot_posterior_hist(results[:, i])
        plot_normal_dist(noise_mean, noise_std, np.amax(n))
        truth_index = i
        plt.axvline(x = truth[truth_index])
        plt.title('param '+str(truth_index))
    plt.savefig(os.path.join(output_dir, "Params_"+epoch_str+".png"))
    plt.show()


def generate(n, generator, noise_mean, noise_std, noise_dim, batch_size):
    print("Generating: ", n, " events...")
    noise = torch.normal(noise_mean, noise_std, size=(batch_size, noise_dim))
    return generator(noise)
