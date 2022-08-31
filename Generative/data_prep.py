import numpy as np
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
from scipy.integrate import quad
from scipy import interpolate
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.special import gamma

# get_u = lambda x, a, b, p: p * x ** a * (1 - x) ** b
# get_d = lambda x, a, b, q: q * x ** a * (1 - x) ** b


u_v_model=lambda x,p: p[0]*x**p[1]*(1-x)**p[2]   

def get_dsigma1(x,p):  
    u=u_v_model(x,p[0:3])
    d=u_v_model(x,p[3:])
    return 4*u+d

def get_dsigma2(x,p):  
    u=u_v_model(x,p[0:3])
    d=u_v_model(x,p[3:])
    return 4*d+u

def gen_events(dsigma,nevents):
    xmin,xmax=0.1,1
    sigma=quad(dsigma,xmin,xmax)[0]
    pdf=lambda x: dsigma(x)/sigma
    get_cdf = lambda x: quad(pdf,x,xmax)[0]
    
    x=np.linspace(xmin,xmax)
    invcdf = interpolate.interp1d([get_cdf(_) for _ in x],x,bounds_error=False,fill_value=0)

    u=np.random.uniform(0,1,nevents)
    events=invcdf(u)
    return events,sigma,pdf

def gen_pdf(sigma):
    xmin,xmax=0.1,1
    norm=quad(sigma,xmin,xmax)[0]
    pdf=lambda x: dsigma(x)/norm

    return pdf

# data generation
def gen_dataset(size, events_per_input, rnd, truth):
    truth_pars = []
    events = []
    for i in range(size):
        #         truth_par = rnd.uniform(0,1,(4,))
        truth_par = truth
        sigma1_events = gen_events(lambda _: get_sigma1(_, truth_par), nevents=events_per_input)[0]
        sigma2_events = gen_events(lambda _: get_sigma2(_, truth_par), nevents=events_per_input)[0]
        events_both = np.concatenate([np.expand_dims(sigma1_events, 1), np.expand_dims(sigma2_events, 1)], axis=1)
        # events_both = np.concatenate([sigma1_events, sigma2_events])
        # np.random.shuffle(events_both)
        events.append(events_both)
        truth_pars.append(truth_par)

    truth_pars = np.asarray(truth_pars)
    events = np.asarray(events)
    return truth_pars, events



# def paramsToEvents(param, parmin, parmax, nevents):
#     nu = tf.gather(param, tf.constant([0]))*(parmax[0] - parmin[0]) + parmin[0]
#     au = tf.gather(param, tf.constant([1]))*(parmax[1] - parmin[1]) + parmin[1]
#     bu = tf.gather(param, tf.constant([2]))*(parmax[2] - parmin[2]) + parmin[2]
#     nd = tf.gather(param, tf.constant([3]))*(parmax[3] - parmin[3]) + parmin[3]
#     ad = tf.gather(param, tf.constant([4]))*(parmax[4] - parmin[4]) + parmin[4]
#     bd = tf.gather(param, tf.constant([5]))*(parmax[5] - parmin[5]) + parmin[5]
#     param = tf.concat([nu, au, bu, nd, ad, bd], axis=0)
#     param = tf.cast(param, tf.float32)
#     tf.print(param)
#     xmin, xmax = 0.1, 0.9999
#     dx = (xmax-xmin)/100
#     x_full_range = tf.range(xmin, xmax, dx)
#     x = tf.linspace(0.1, 0.9999, 100)
    
# #     nevents = 100000

#     def get_ud(x, p):
#         tf.print("x, p", x, p)
#         u = p[2]*tf.math.pow(x, p[0]) * tf.math.pow((1-x), p[1])
#         d = p[5]*tf.math.pow(x, p[3]) * tf.math.pow((1-x), p[4])
#         return u, d

#     def integral_approx(y, dx):
#         return tf.reduce_sum(y[:-1]) * dx
    
#     def interpolate( dx_T, dy_T, x):
#         delVals = dx_T - x
#         ind_1   = tf.argmin(tf.sign( delVals )*delVals)
#         ind_0   = ind_1 - 1


#         value   = tf.cond( x[0] <= dx_T[0], 
#                           lambda : dy_T[:1], 
#                           lambda : tf.cond( 
#                                  x[0] >= dx_T[-1], 
#                                  lambda : dy_T[-1:],
#                                  lambda : (dy_T[ind_0] +                \
#                                            (dy_T[ind_1] - dy_T[ind_0])  \
#                                            *(x-dx_T[ind_0])/            \
#                                            (dx_T[ind_1]-dx_T[ind_0]))
#                          ))
# #                 tf.print(ind_1, ind_0, dy_T[ind_1], dy_T[ind_0], x, dx_T[ind_1], dx_T[ind_0])
#         result = tf.multiply(value[0], 1)

#         return result

#     def inverse_cdf(cdf_allx1):

#         u = tf.random.uniform(shape=(1*nevents,), minval=0.0, maxval=0.9999)
#         cdf_sort_indx1 = tf.argsort(cdf_allx1)
#         cdf_sort1 = tf.gather(cdf_allx1, cdf_sort_indx1)
#         x_sort1 = tf.gather(x, cdf_sort_indx1)

#         cdf_sort1 = tf.expand_dims(cdf_sort1, 0)
#         cdf_sort1 = tf.expand_dims(cdf_sort1, 2)
#         x_sort1 = tf.expand_dims(x_sort1, 0)
#         x_sort1 = tf.expand_dims(x_sort1, 2)
#         tf.print("sorted_x: ", x_sort1)
#         tf.print("min max cdf: ", tf.reduce_min(cdf_sort1), tf.reduce_max(cdf_sort1))
#         u = tf.expand_dims(u, 0)
#         u = tf.expand_dims(u, 2)
#         events_out1 = tfa.image.interpolate_spline(cdf_sort1, x_sort1, u, order=0)
# #         events_out1 = interpolate(cdf_allx1, x_full_range, u)
#         tf.print(tf.reduce_sum(events_out1))
#         events_out1 = tf.squeeze(events_out1)
#         events_out1 = tf.reshape(events_out1, (nevents,))
#         return events_out1


#     def gen_events_sigmas(true_params):

#         u_full, d_full = get_ud(x_full_range, true_params)
#         sigma1 = 4*u_full+d_full
#         sigma2 = 4*d_full+u_full
# #         print(sigma1)
#         norm1 = integral_approx(sigma1, dx)
#         norm2 = integral_approx(sigma2, dx)
# #         print(norm1, norm2)
#         u, d = get_ud(x, true_params)
#         tf.print("u: ", tf.reduce_sum(u))
#         pdf1 = ((4*u)+d)/norm1
#         pdf2 = ((4*d)+u)/norm2
#         tf.print("PDF1: ", pdf1)
#         cdf_allx1 = (tf.math.cumsum(pdf1[::-1]*dx)/tf.math.reduce_sum(pdf1*dx))[::-1]
#         cdf_allx2 = (tf.math.cumsum(pdf2[::-1]*dx)/tf.math.reduce_sum(pdf2*dx))[::-1]

#         events1 = inverse_cdf(cdf_allx1)
#         events2 = inverse_cdf(cdf_allx2)

#         events = tf.concat([tf.expand_dims(events1, 0), tf.expand_dims(events2, 0)], 0)

#         return events, norm1, norm2

#     events, norm1, norm2 = gen_events_sigmas(param)
#     events = (events - 0.1)/0.9
#     return events, norm1, norm2

# @tf.function
# def wrapperGenerator(params, parmin, parmax, nevents):
    
#     return tf.vectorized_map(lambda x:paramsToEvents(x, parmin, parmax, nevents), params)


# def paramsToNorm(param, nevents):
#     a = (tf.gather(param, tf.constant([0]))-1.)/2.
#     b = tf.gather(param, tf.constant([1]))*3.
# #         p = tf.gather(param, tf.constant([2]))
#     p = tf.expand_dims(tf.constant(0.1), 0)
#     c = (tf.gather(param, tf.constant([2]))-1.)/2.
#     d = tf.gather(param, tf.constant([3]))*3
# #         q = tf.gather(param, tf.constant([5]))
#     q = tf.expand_dims(tf.constant(0.85), 0)
#     param = tf.concat([a, b, p, c, d, q], axis=0)
#     xmin, xmax = 0.1, 0.9999
#     dx = 0.01
#     x_full_range = tf.range(xmin, xmax, dx)
#     x = tf.linspace(0.1, 1, 50)
# #     nevents = 100000

#     def get_ud(x, p):
#         u = p[2]*tf.math.pow(x, p[0]) * tf.math.pow((1-x), p[1])
#         d = p[5]*tf.math.pow(x, p[3]) * tf.math.pow((1-x), p[4])
#         return u, d

#     def integral_approx(y, dx):
#         return tf.reduce_sum(y[:-1]) * dx

#     def inverse_cdf(cdf_allx1):

#         u = tf.random.uniform(shape=(1*nevents,), minval=0.0, maxval=1.0)

#         cdf_sort_indx1 = tf.argsort(cdf_allx1)
#         cdf_sort1 = tf.gather(cdf_allx1, cdf_sort_indx1)
#         x_sort1 = tf.gather(x, cdf_sort_indx1)

#         cdf_sort1 = tf.expand_dims(cdf_sort1, 0)
#         cdf_sort1 = tf.expand_dims(cdf_sort1, 2)
#         x_sort1 = tf.expand_dims(x_sort1, 0)
#         x_sort1 = tf.expand_dims(x_sort1, 2)
#         u = tf.expand_dims(u, 0)
#         u = tf.expand_dims(u, 2)
#         events_out1 = tfa.image.interpolate_spline(cdf_sort1, x_sort1, u, order=1)
#         events_out1 = tf.squeeze(events_out1)
#         events_out1 = tf.reshape(events_out1, (nevents,))
#         return events_out1


#     def gen_events_sigmas(true_params):

#         u_full, d_full = get_ud(x_full_range, true_params)
#         sigma1 = 4*u_full+d_full
#         sigma2 = 4*d_full+u_full
#         norm1 = integral_approx(sigma1, dx)
#         norm2 = integral_approx(sigma2, dx)

#         return norm1, norm2

#     norm1, norm2 = gen_events_sigmas(param)
#     return norm1, norm2


# @tf.function
# def wrapperGenerator_param(params, nevents):
    
#     return tf.vectorized_map(lambda x:paramsToNorm(x, nevents), params)

from scipy import interpolate
import tensorflow as tf
import tensorflow_addons as tfa

def paramsToEvents(param, parmin, parmax, nevents):
    nu = tf.gather(param, tf.constant([0]))*(parmax[0] - parmin[0]) + parmin[0]
    au = tf.gather(param, tf.constant([1]))*(parmax[1] - parmin[1]) + parmin[1]
    bu = tf.gather(param, tf.constant([2]))*(parmax[2] - parmin[2]) + parmin[2]
    nd = tf.gather(param, tf.constant([3]))*(parmax[3] - parmin[3]) + parmin[3]
    ad = tf.gather(param, tf.constant([4]))*(parmax[4] - parmin[4]) + parmin[4]
    bd = tf.gather(param, tf.constant([5]))*(parmax[5] - parmin[5]) + parmin[5]
    param = tf.concat([nu, au, bu, nd, ad, bd], axis=0)
    param = tf.cast(param, tf.float32)

    
    xmin, xmax = 0.1, 0.99999
    dx = (xmax-xmin)/1000
    x_full_range = tf.range(xmin, xmax, dx)
#     x = tf.linspace(0.1, 0.9999, 10)
    
    def get_ud(x, p):
        u = p[0]*tf.math.pow(x, p[1]) * tf.math.pow((1-x), p[2])
        d = p[3]*tf.math.pow(x, p[4]) * tf.math.pow((1-x), p[5])
        return u, d

    def integral_approx(y, dx):
        return tf.reduce_sum(y[:-1]) * dx

    def tf_interp(x, xs, ys):
        # determine the output data type
        ys = tf.convert_to_tensor(ys)
        dtype = ys.dtype

        # normalize data types
        ys = tf.cast(ys, tf.float64)
        xs = tf.cast(xs, tf.float64)
        x = tf.cast(x, tf.float64)

        # pad control points for extrapolation
        xs = tf.concat([[xs.dtype.min], xs, [xs.dtype.max]], axis=0)
        ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

        # compute slopes, pad at the edges to flatten
        ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        ms = tf.pad(ms[:-1], [(1, 1)])

        # solve for intercepts
        bs = ys - ms*xs

        # search for the line parameters at each input data point
        # create a grid of the inputs and piece breakpoints for thresholding
        # rely on argmax stopping on the first true when there are duplicates,
        # which gives us an index into the parameter vectors
        i = tf.math.argmax(xs[..., tf.newaxis, :] > x[..., tf.newaxis], axis=-1)
        m = tf.gather(ms, i, axis=-1)
        b = tf.gather(bs, i, axis=-1)

        # apply the linear mapping at each input data point
        y = m*x + b
        return tf.cast(tf.reshape(y, tf.shape(x)), dtype)




    def inverse_cdf(cdf_allx1):
        u = tf.random.uniform(shape=(1*nevents,), minval=0.0, maxval=0.9999)
        cdf_sort_indx1 = tf.argsort(cdf_allx1)
        cdf_sort1 = tf.gather(cdf_allx1, cdf_sort_indx1)
        x_sort1 = tf.gather(x_full_range, cdf_sort_indx1)
        
#         cdf_sort1 = tf.expand_dims(cdf_sort1, 0)
#         cdf_sort1 = tf.expand_dims(cdf_sort1, 2)
#         x_sort1 = tf.expand_dims(x_sort1, 0)
#         x_sort1 = tf.expand_dims(x_sort1, 2)
        
#         u = tf.expand_dims(u, 0)
#         u = tf.expand_dims(u, 2)
#         tf.print("U: ", u)
#         tf.print("cdf_sort1: ", cdf_sort1)
#         tf.print("x_sort1: ", x_sort1)
#         events_out1 = tfa.image.interpolate_spline(cdf_sort1, x_sort1, u, order=1)
#         tf.print("events: ", tf.reduce_max(events_out1), tf.reduce_min(events_out1))
#         tf.print("events: ", events_out1)
#         events_out1 = tf.squeeze(events_out1)
#         events_out1 = tf.reshape(events_out1, (nevents,))
#         events_out1 = interpolate(cdf_allx1, x_full_range, u)
#         events_out1 = interpolate.interp1d(cdf_sort1, x_sort1)(u)
#         events_out1 = tf.numpy_function(interpolate.interp1d(cdf_sort1, x_sort1), [u], tf.float32, stateful=True, name=None)
        events_out1 = tf_interp(u, cdf_sort1, x_sort1)
        events_out1 = tf.reshape(events_out1, (nevents,))
        return events_out1


    def gen_events_sigmas(true_params):

        u_full, d_full = get_ud(x_full_range, true_params)
        sigma1 = 4*u_full+d_full
        sigma2 = 4*d_full+u_full
        
        norm1 = integral_approx(sigma1, dx)
        norm2 = integral_approx(sigma2, dx)

        u, d = get_ud(x_full_range, true_params)
        
        
        pdf1 = ((4*u)+d)/norm1
        pdf2 = ((4*d)+u)/norm2

        cdf_allx1 = (tf.math.cumsum(pdf1[::-1]*dx)/tf.math.reduce_sum(pdf1*dx))[::-1]
        cdf_allx2 = (tf.math.cumsum(pdf2[::-1]*dx)/tf.math.reduce_sum(pdf2*dx))[::-1]

        events1 = inverse_cdf(cdf_allx1)
        events2 = inverse_cdf(cdf_allx2)

        events = tf.concat([tf.expand_dims(events1, 0), tf.expand_dims(events2, 0)], 0)
#         tf.print("before: ", events1, events2)
#         events = tf.concat([events1, events2], 0)
#         tf.print("events: ", events)
        return events, norm1, norm2
    
    events, norm1, norm2 = gen_events_sigmas(param)
#     print(events, norm1, norm2)
#     events = (events - 0.1)/0.9
    return events, norm1, norm2

@tf.function
def wrapperGenerator(params, parmin, parmax, nevents):
    
    return tf.vectorized_map(lambda x:paramsToEvents(x, parmin, parmax, nevents), params)