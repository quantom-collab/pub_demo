import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization, Lambda
import sys
import random
seed = 22
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.colors import LogNorm
from scipy.integrate import quad
import tensorflow.keras.backend as K

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
from scipy.special import gamma
from data_prep import u_v_model, gen_events, get_dsigma1, get_dsigma2, wrapperGenerator
from GAN import GAN

# outputDirectory = "/work/data_science/kishan/Theory/Experiments_0/"
beta=lambda a,b: gamma(a)*gamma(b)/gamma(a+b)

au=-0.5; bu=3; nu=2/beta(au+1,bu+1)
ad=-0.5; bd=4; nd=1/beta(au+1,bu+1)
truth_par=np.array([nu,au,bu,nd,ad,bd])
print("Ground Truth parameters: ", truth_par)


outputDirectory = sys.argv[1]
os.makedirs(outputDirectory, exist_ok=True)

NUMOFEVENTS = 100000
parmin = np.array([0.0, -1.0, 0.0, 0.0, -1.0, 0.0])
parmax = np.array([3.0, 1.0, 5.0, 3.0, 1.0, 5.0])
sigma1_events, sigma1_norm, sigma1_pdf = gen_events(lambda _: get_dsigma1(_,truth_par),nevents=NUMOFEVENTS)
sigma1_events = sigma1_events.reshape(-1,1)
sigma2_events, sigma2_norm, sigma2_pdf = gen_events(lambda _: get_dsigma2(_,truth_par),nevents=NUMOFEVENTS)
sigma2_events = sigma2_events.reshape(-1,1)
print(sigma1_norm, sigma2_norm)
sigma1_norm = np.repeat(sigma1_norm, NUMOFEVENTS)
sigma1_norm = sigma1_norm.reshape(-1,1)
sigma2_norm = np.repeat(sigma2_norm, NUMOFEVENTS)
sigma2_norm = sigma2_norm.reshape(-1,1)
evts = np.concatenate([sigma1_events, sigma2_events], axis = 1)

plt.clf()
plt.hist(evts[:, 0], bins=50, alpha=0.5, label="sigma1")
plt.hist(evts[:, 1], bins=50, alpha=0.5, label="sigma2")
plt.title("Ground truth events distribution (Scipy)")
plt.legend()
plt.savefig(os.path.join(outputDirectory, "GT_eventDistribution.png"))
plt.show()


truth = np.array([[2.1875, -0.5, 3, 1.09375, -0.5, 4],
                 [2.1875, -0.5, 3, 1.09375, -0.5, 4]]).astype(np.float32)
parmin = np.array([0.0, -1.0, 0.0, 0.0, -1.0, 0.0])
parmax = np.array([3.0, 1.0, 5.0, 3.0, 1.0, 5.0])

def normalize_pars(t, parmin, parmax):
    return (t - parmin) / (parmax - parmin)

truth_scaled = normalize_pars(truth, parmin, parmax)
train_events, train_norm1, train_norm2 = wrapperGenerator(truth_scaled, parmin, parmax, 100000)
train_events = train_events[0]
train_norm1 = train_norm1[0]
train_norm2 = train_norm2[0]

plt.clf()
plt.hist(train_events[0].numpy(), bins=50, alpha=0.5, label="sigma1")
plt.hist(train_events[1].numpy(), bins=50, alpha=0.5, label="sigma2")
plt.title("Ground truth events distribution (TF)")
plt.legend()
plt.savefig(os.path.join(outputDirectory, "GT_TF_eventDistribution.png"))
plt.show()

# my_strategy = tf.distribute.MirroredStrategy()
# with my_strategy.scope():
model = GAN(train_events.numpy(), train_norm1.numpy(), train_norm2.numpy(), batch_size=256, truth=truth_scaled[0], parmin=parmin, parmax=parmax, model_load_Path=None, start_epoch=0, output_directory=outputDirectory)
    
model.train(400000, sample_interval=1000)


