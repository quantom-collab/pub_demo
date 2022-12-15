import torch
import torch.nn as nn
import torch.optim as optim
from data_prep_torch import wrapperGenerator

class Generator(nn.Module):
    def __init__(self, input_size, output_size=6, n_units=32, use_bias=True):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),

            nn.Linear(n_units, output_size, bias=use_bias),
            nn.Sigmoid()
        )
       #self.main = nn.Sequential(
       #    nn.Linear(input_size, output_size, bias=use_bias),
       #    nn.Sigmoid()
       #)

    def forward(self, input):
        return self.main(input)


class Generator2(nn.Module):
    def __init__(self, input_size, output_size=6, n_units=256, use_bias=True, input_mean=0.0, input_std=1.0):
        super().__init__()
        self.input_mean = input_mean
        self.input_std  = input_std
        self.main = nn.Sequential(
            nn.Linear(input_size, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),

            nn.Linear(n_units, n_units, bias=False),
            nn.LeakyReLU(0.2),

            nn.BatchNorm1d(n_units),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),

            nn.Linear(n_units, n_units, bias=False),
            nn.LeakyReLU(0.2),

            nn.BatchNorm1d(n_units),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),

            nn.Linear(n_units, n_units, bias=False),
            nn.LeakyReLU(0.2),

            nn.BatchNorm1d(n_units),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),

            nn.Linear(n_units, output_size, bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main((x - self.input_mean)/self.input_std)


class Discriminator(nn.Module):
    def __init__(self, input_size, n_units=256, dropout_rate=0.0, use_bias=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

#           nn.Linear(n_units, n_units, bias=use_bias),
#           nn.LeakyReLU(0.2),
#           nn.Dropout(dropout_rate),

#           nn.Linear(n_units, n_units, bias=use_bias),
#           nn.LeakyReLU(0.2),
#           nn.Dropout(dropout_rate),

#           nn.Linear(n_units, n_units, bias=use_bias),
#           nn.LeakyReLU(0.2),
#           nn.Dropout(dropout_rate),

            nn.Linear(n_units, 1, bias=use_bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator2(nn.Module):
    def __init__(self, input_size, n_units=256, dropout_rate=0.0, use_bias=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(n_units, n_units, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.BatchNorm1d(n_units),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(n_units, n_units, bias=use_bias),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

#           nn.Linear(n_units, n_units, bias=use_bias),
#           nn.LeakyReLU(0.2),
#           nn.Dropout(dropout_rate),

#           nn.Linear(n_units, n_units, bias=use_bias),
#           nn.LeakyReLU(0.2),
#           nn.Dropout(dropout_rate),

#           nn.Linear(n_units, n_units, bias=use_bias),
#           nn.LeakyReLU(0.2),
#           nn.Dropout(dropout_rate),
        )
        self.layer_mu  = nn.Linear(n_units, 1)
        self.layer_std = nn.Linear(n_units, 1)

    def forward(self, x):
        y = self.main(x)
        y_mu  = self.layer_mu(y)
        y_std = self.layer_std(y)
        out_mu  = torch.sigmoid(y_mu)
        out_std = torch.sigmoid(y_std)
        return [out_mu, out_std]


def generator_weights_init(m):
    ''' Custom weights initialization called on Generator '''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        #nn.init.eye_(m.weight.data)
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def disciminator_weights_init(m):
    ''' Custom weights initialization called on Discriminator '''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def params_prior(generated_params, prior_mean, prior_std):
    return torch.mean( 0.5 * (generated_params - prior_mean)**2 / prior_std**2 )


class GAN():
    def __init__(self, train_events, train_norm1, train_norm2,
                 noise_mean=0.5,
                 noise_std=0.18,
                 params_dim=6,
                 events_dim=2,
                 events_sim=1,
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
        self.events_sim = events_sim

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
        self.generator.apply(generator_weights_init)
        self.discriminator.apply(disciminator_weights_init)
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

    def train(self, n_epochs=10+1, plot_interval=10):
        label_obs = torch.zeros((self.batch_size,))
        label_gen = torch.zeros((self.batch_size * self.events_sim,))

        print("Starting Training Loop...")
        for epoch in range(n_epochs):
            indices = torch.randint(0, self.train_events.shape[0], (self.batch_size,))
            events_real = self.train_events[indices,:]

            # train discriminator
            for d_step in range(self.d_n_steps):
                noise = torch.normal(self.noise_mean, self.noise_std, size=(self.batch_size, self.noise_dim))
                self.d_optimizer.zero_grad()

                label_obs.fill_(1.0)
                real_pred = self.discriminator(events_real).view(-1)
                d_loss_real = 0.5 * self.d_loss_function(label_obs, real_pred)

                params = self.generator(noise)
                events_fake, norm1, norm2 = wrapperGenerator(params, self.parmin, self.parmax, self.events_sim)
                events_fake = torch.reshape(torch.transpose(events_fake, 1, 2), (-1,self.events_dim))
                label_gen.fill_(0.0)
                fake_pred = self.discriminator(events_fake).view(-1)
                d_loss_fake = 0.5 * self.d_loss_function(label_gen, fake_pred)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

            # train generator
            for g_step in range(self.g_n_steps):
                noise = torch.normal(self.noise_mean, self.noise_std, size=(self.batch_size, self.noise_dim))
                self.g_optimizer.zero_grad()
                params = self.generator(noise)
                events_fake, norm1, norm2 = wrapperGenerator(params, self.parmin, self.parmax, self.events_sim)
                events_fake = torch.reshape(torch.transpose(events_fake, 1, 2), (-1,self.events_dim))
                label_gen.fill_(1.0)
                fake_pred = self.discriminator(events_fake).view(-1)
                g_loss_fake = self.g_loss_function(label_gen, fake_pred)
                g_prior_term = 0.1 * params_prior(params, self.noise_mean, self.noise_std)

                g_loss_norm = self.norm1_loss_function(self.train_norms1, norm1) + \
                              self.norm2_loss_function(self.train_norms2, norm2)

                g_loss = g_loss_fake + g_loss_norm + g_prior_term
                g_loss.backward()
                self.g_optimizer.step()

            # print status
            if epoch % 100 == 0:
                print("Epoch:", epoch,
                      "; norms:", torch.mean(norm1).item(), torch.mean(norm2).item(),
                      "; g_loss:", g_loss.item(), " d_loss:", d_loss.item())
            if epoch % plot_interval == 0:
                with torch.no_grad():
                    plot_events_and_params(epoch, self.generator,
                                           self.train_events, self.train_norms1, self.train_norms2,
                                           self.noise_mean, self.noise_std, self.noise_dim,
                                           100*self.events_sim, self.batch_size,
                                           self.truth, self.parmin, self.parmax,
                                           self.output_dir)
                #self.d_n_steps += 1
                #self.d_n_steps = min(self.d_n_steps, 5)


class WGAN():
    def __init__(self, train_events, train_norm1, train_norm2,
                 noise_mean=0.5,
                 noise_std=0.18,
                 params_dim=6,
                 events_dim=2,
                 events_sim=1,
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
        self.events_sim = events_sim

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
        self.generator.apply(generator_weights_init)
        self.discriminator.apply(disciminator_weights_init)
        # print models
        print(self.generator)
        print(self.discriminator)

        # set beta1 hyperparameter for Adam optimizer
        opt_beta1 = 0.5
        opt_beta2 = 0.999
        # create optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(),     lr=learning_rate, betas=(opt_beta1, opt_beta2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(opt_beta1, opt_beta2))
        #self.g_optimizer = optim.RMSprop(self.generator.parameters(),     lr=learning_rate)
        #self.d_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=learning_rate)
        self.g_loss_function = self.generator_loss
        self.d_loss_function = self.discriminator_loss
        self.norm1_loss_function = nn.MSELoss()
        self.norm2_loss_function = nn.MSELoss()

    def generator_loss(self, fake_events):
        return -torch.mean(fake_events)

    def discriminator_loss(self, real_events, fake_events):
        real_loss = torch.mean(real_events)
        fake_loss = torch.mean(fake_events)
        return fake_loss - real_loss

    def d_clip_gradient(self, clip_magn=1.0):
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), clip_magn, norm_type=2.0, error_if_nonfinite=False)

    def train(self, n_epochs=10+1, plot_interval=10):
        print("Starting Training Loop...")
        for epoch in range(n_epochs):
            indices = torch.randint(0, self.train_events.shape[0], (self.batch_size,))
            events_real = self.train_events[indices,:]

            # train discriminator
            for d_step in range(self.d_n_steps):
                noise = torch.normal(self.noise_mean, self.noise_std, size=(self.batch_size, self.noise_dim))
                self.d_optimizer.zero_grad()

                real_pred = self.discriminator(events_real).view(-1)

                params = self.generator(noise)
                events_fake, norm1, norm2 = wrapperGenerator(params, self.parmin, self.parmax, self.events_sim)
                events_fake = torch.reshape(torch.transpose(events_fake, 1, 2), (-1,self.events_dim))
                fake_pred = self.discriminator(events_fake).view(-1)

                d_loss = self.d_loss_function(real_pred, fake_pred)
                d_loss.backward()
                self.d_clip_gradient()
                self.d_optimizer.step()

            # train generator
            for g_step in range(self.g_n_steps):
                noise = torch.normal(self.noise_mean, self.noise_std, size=(self.batch_size, self.noise_dim))
                self.g_optimizer.zero_grad()
                params = self.generator(noise)
                events_fake, norm1, norm2 = wrapperGenerator(params, self.parmin, self.parmax, self.events_sim)
                events_fake = torch.reshape(torch.transpose(events_fake, 1, 2), (-1,self.events_dim))
                fake_pred = self.discriminator(events_fake).view(-1)

                g_loss_fake = self.g_loss_function(fake_pred)
                g_loss_norm = self.norm1_loss_function(self.train_norms1, norm1) + \
                              self.norm2_loss_function(self.train_norms2, norm2)
                #g_loss_prior = 0.1 * params_prior(params, self.noise_mean, self.noise_std)

                g_loss = g_loss_fake + g_loss_norm #+ g_loss_prior
                g_loss.backward()
                self.g_optimizer.step()

            # print status
            if epoch % 100 == 0:
                print("Epoch:", epoch,
                      "; norms:", torch.mean(norm1).item(), torch.mean(norm2).item(),
                      "; g_loss:", g_loss.item(), " d_loss:", d_loss.item())
            if epoch % plot_interval == 0:
                with torch.no_grad():
                    plot_events_and_params(epoch, self.generator,
                                           self.train_events, self.train_norms1, self.train_norms2,
                                           self.noise_mean, self.noise_std, self.noise_dim,
                                           100*self.events_sim, self.batch_size,
                                           self.truth, self.parmin, self.parmax,
                                           self.output_dir)
                #self.d_n_steps += 1
                #self.d_n_steps = min(self.d_n_steps, 5)


class GAN2():
    def __init__(self, train_events, train_norm1, train_norm2,
                 noise_mean=0.5,
                 noise_std=0.2,
                 params_dim=6,
                 events_dim=2,
                 events_sim=1,
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
        self.events_sim = events_sim

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.d_n_steps = 1
        self.g_n_steps = 1

        self.truth  = truth
        self.parmin = parmin
        self.parmax = parmax
        self.output_dir = output_dir

        # create generator and discriminator models
        self.generator     = Generator2(self.params_dim, input_mean=self.noise_mean, input_std=self.noise_std)
        self.discriminator = Discriminator2(self.events_dim)
        # initialize model weights
        self.generator.apply(generator_weights_init)
        self.discriminator.apply(disciminator_weights_init)
        # print models
        print(self.generator)
        print(self.discriminator)

        # set beta1 hyperparameter for Adam optimizer
        opt_beta1 = 0.5
        opt_beta2 = 0.999
        # create optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(),     lr=learning_rate, betas=(opt_beta1, opt_beta2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(opt_beta1, opt_beta2))
        self.g_loss_function = self.generator_loss
        self.d_loss_function = self.discriminator_loss
        self.norm1_loss_function = nn.MSELoss()
        self.norm2_loss_function = nn.MSELoss()
        self.param_prior_weight     = 1.0
        self.param_prior_weight_min = 1.0e-6

    def misfit_loss(self, real_events_mu, real_events_std, fake_events_mu, fake_events_std):
        if real_events_mu is not None and real_events_std is not None:
            real_mu_loss  = torch.mean((real_events_mu - 1.0)**2)
            #real_std_loss = torch.mean(torch.log(real_events_std))
        else:
            real_mu_loss  = 0.0
            #real_std_loss = 0.0
        if fake_events_mu is not None and fake_events_std is not None:
            fake_mu_loss  = torch.mean(fake_events_mu**2)
            #fake_std_loss = torch.mean(torch.log(fake_events_std))
        else:
            fake_mu_loss  = 0.0
            #fake_std_loss = 0.0
        return real_mu_loss + fake_mu_loss

    def param_loss(self, params):
        if params is not None:
            #return self.param_prior_weight * torch.mean( 0.25 * (params - self.noise_mean)**4 / self.noise_std**4 )
                   #0.01*( torch.mean(-torch.log(params)) + torch.mean(-torch.log(1.0 - params)) )
            return 0.0
        else:
            return 0.0

    def generator_loss(self, fake_events_mu, fake_events_std, params):
        mloss = self.misfit_loss(fake_events_mu, fake_events_std, None, None)
        ploss = self.param_loss(params)
        return mloss + ploss

    def discriminator_loss(self, real_events_mu, real_events_std, fake_events_mu, fake_events_std, params):
        mloss = self.misfit_loss(real_events_mu, real_events_std, fake_events_mu, fake_events_std)
        #ploss = self.param_loss(params)
        return mloss #+ ploss

    def d_clip_gradient(self, clip_magn=1.0):
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), clip_magn, norm_type=2.0, error_if_nonfinite=False)

    def train(self, n_epochs=10+1, plot_interval=10):
        print("Starting Training Loop...")
        for epoch in range(n_epochs):
            indices = torch.randint(0, self.train_events.shape[0], (self.batch_size,))
            events_real = self.train_events[indices,:]

            # train discriminator
            for d_step in range(self.d_n_steps):
                g_noise = torch.normal(self.noise_mean, self.noise_std, size=(self.batch_size, self.noise_dim))
                g_noise.clamp_(0.0, 1.0)
                self.d_optimizer.zero_grad()

                real_pred_mu, real_pred_std = self.discriminator(events_real)
                #real_pred_mu = real_pred_mu + torch.normal(0.0, real_pred_std)

                params = self.generator(g_noise)
                events_fake, norm1, norm2 = wrapperGenerator(params, self.parmin, self.parmax, self.events_sim)
                events_fake = torch.reshape(torch.transpose(events_fake, 1, 2), (-1,self.events_dim))
                fake_pred_mu, fake_pred_std = self.discriminator(events_fake)
                #fake_pred_mu = fake_pred_mu + torch.normal(0.0, fake_pred_std)

                d_loss = self.d_loss_function(real_pred_mu, real_pred_std, fake_pred_mu, fake_pred_std, params)
                d_loss.backward()
                self.d_clip_gradient()
                self.d_optimizer.step()

            # train generator
            for g_step in range(self.g_n_steps):
                g_noise = torch.normal(self.noise_mean, self.noise_std, size=(self.batch_size, self.noise_dim))
                g_noise.clamp_(0.0, 1.0)
                self.g_optimizer.zero_grad()
                params = self.generator(g_noise)
                events_fake, norm1, norm2 = wrapperGenerator(params, self.parmin, self.parmax, self.events_sim)
                events_fake = torch.reshape(torch.transpose(events_fake, 1, 2), (-1,self.events_dim))
                fake_pred_mu, fake_pred_std = self.discriminator(events_fake)
                #fake_pred_mu = fake_pred_mu + torch.normal(0.0, fake_pred_std)

                g_loss_fake = self.g_loss_function(fake_pred_mu, fake_pred_std, params)
                g_loss_norm = self.norm1_loss_function(self.train_norms1, norm1) + \
                              self.norm2_loss_function(self.train_norms2, norm2)

                g_loss = g_loss_fake + g_loss_norm
                g_loss.backward()
                self.g_optimizer.step()

            # print status
            if epoch % 100 == 0:
                with torch.no_grad():
                    print("Epoch:", epoch,
                          "; norms:", torch.mean(norm1).item(), torch.mean(norm2).item(),
                          "; g_loss:", g_loss.item(), " d_loss:", d_loss.item(),
                          "; std min & max:", torch.min(fake_pred_std).item(), torch.max(fake_pred_std).item(),
                          "; params abs err:", torch.abs(self.truth - torch.mean(params, dim=0)).numpy()
                    )
                self.param_prior_weight = max(self.param_prior_weight_min, 0.5*self.param_prior_weight)
            if epoch % plot_interval == 0:
                with torch.no_grad():
                    plot_events_and_params(epoch, self.generator,
                                           self.train_events, self.train_norms1, self.train_norms2,
                                           self.noise_mean, self.noise_std, self.noise_dim,
                                           100*self.events_sim, self.batch_size,
                                           self.truth, self.parmin, self.parmax,
                                           self.output_dir)
                #self.d_n_steps += 1
                #self.d_n_steps = min(self.d_n_steps, 5)


def plot_events_and_params(epoch, generator,
                           train_events, train_norms1, train_norms2,
                           noise_mean, noise_std, noise_dim, events_sim, batch_size,
                           truth, parmin, parmax,
                           output_dir):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("Inside sample interval block...")

    #generator.save_weights(os.path.join(output_dir, "2022.08.21_generator_6Params.h5"))
    #discriminator.save_weights(os.path.join(output_dir, "2022.08.21_discriminator_6Params.h5"))
    #print("models saved...")

    gen_noise, gen_params = generate(generator, noise_mean, noise_std, noise_dim, batch_size)
    gen_params_m = torch.mean(gen_params, dim=0, keepdim=True)
    gen_events, _, _ = wrapperGenerator(gen_params, parmin, parmax, events_sim)
    gen_events = gen_events.numpy()
    gen_events = np.reshape(np.transpose(gen_events, axes=(0,2,1)), (-1,gen_events.shape[2]))
    print('gen_params_m', gen_params_m.shape)
    gen_events_m, _, _ = wrapperGenerator(gen_params_m, parmin, parmax, events_sim)
    print('gen_events_m', gen_events_m.shape)
    gen_events_m = gen_events_m.numpy()
    print("Events generated...")
    print("training norms", train_norms1[0], train_norms2[0])
    plt.clf()
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,2,1)
    ax1.hist([gen_events_m[0,0,:],gen_events[:,0],train_events[:events_sim,0]],
             bins=100, histtype='step', density=True, alpha=0.7, linewidth=2,
             color=['royalblue','orange','green'],
             label=['sigma1: GAN param_mean','sigma1: GAN all params','sigma1: Sim']
    )
    ax1.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.hist([gen_events_m[0,1,:],gen_events[:,1],train_events[:events_sim,1]],
             bins=100, histtype='step', density=True, alpha=0.7, linewidth=2,
             color=['royalblue','orange','green'],
             label=['sigma2: GAN param_mean','sigma2: GAN all params','sigma2: Sim']
    )
    ax2.legend()
    plt.title("At Epoch - "+str(epoch))
    epoch_str = str(epoch).zfill(6)
    plt.savefig(os.path.join(output_dir, "Events_"+epoch_str+".png"))
    plt.show()

    noise_np  = gen_noise.numpy()
    params_np = gen_params.numpy()
    n_results = params_np.shape[1]

    def plot_posterior_hist(input_noise, generated_params):
        n,_,_ = plt.hist([generated_params,input_noise],
                         bins=100, range=(0.0,1.0), histtype='step', density=True, alpha=0.7, linewidth=2,
                         color=['royalblue','gray'],
                         label=['posterior','prior']
        )
        return n

    def plot_normal_dist(mean, std, scale=None):
        if scale is None:
            scale = 1.0/(std*np.sqrt(2*np.pi))
        x = np.linspace(0, 1, 128)
        y = scale * np.exp(-(x - mean)**2/(2*std)**2)
        plt.plot(x, y, color='gray')

    plt.figure(figsize=(20,15))
    for i in range(n_results):
        plt.subplot(n_results, 1, i+1)
        n = plot_posterior_hist(noise_np[:,i], params_np[:,i])
        plot_normal_dist(noise_mean, noise_std, np.amax(n))
        plt.axvline(x=truth[i], color='green', linewidth=2)
        plt.title('param '+str(i))

    plt.figure(figsize=(20,15))
    column_labels = ['param_{}'.format(i) for i in range(n_results)]
    results_df = pd.DataFrame(data=params_np, columns=column_labels)
    g = sns.pairplot(results_df, corner=False,
                     kind='scatter', plot_kws={'alpha': 0.5},
                     diag_kind='kde' #, diag_kws={'bins':20, 'binrange':(0.0,1.0)}
    )
    for i in range(n_results):
        for j in range(n_results):
            g.axes[i,j].set_xlim((0.0, 1.0))
            g.axes[i,j].grid()
            if i != j:
                g.axes[i,j].set_ylim((0.0, 1.0))
    g.map_offdiag(sns.kdeplot, levels=4, color='purple')
    g.map_offdiag(sns.regplot, scatter=False, color='purple', line_kws={'linestyle':'dashed'})
    for i in range(n_results):
        for j in range(n_results):
            if i == j:
                g.axes[i,j].axvline(x=truth[i], color='green')
            else:
                g.axes[i,j].scatter(truth[j], truth[i], s=16**2, marker='+', color='darkgreen', linewidth=4)

    plt.savefig(os.path.join(output_dir, "Params_"+epoch_str+".png"))
    plt.show()


def generate(generator, noise_mean, noise_std, noise_dim, batch_size):
    noise = torch.normal(noise_mean, noise_std, size=(batch_size, noise_dim))
    noise.clamp_(0.0, 1.0)
    return noise, generator(noise)
