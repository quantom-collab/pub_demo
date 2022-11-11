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
import torch
from scipy.special import gamma
from scipy import interpolate
import functorch


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



def paramsToEvents(param, parmin, parmax, nevents):
#     nu = param[0]*(parmax[0] - parmin[0]) + parmin[0]
#     au = param[1]*(parmax[1] - parmin[1]) + parmin[1]
#     bu = param[2]*(parmax[2] - parmin[2]) + parmin[2]
#     nd = param[3]*(parmax[3] - parmin[3]) + parmin[3]
#     ad = param[4]*(parmax[4] - parmin[4]) + parmin[4]
#     bd = param[5]*(parmax[5] - parmin[5]) + parmin[5]
#     param = tf.concat([nu, au, bu, nd, ad, bd], axis=0)
#     param = tf.cast(param, tf.float32)
    param = param * (parmax - parmin) + parmin

    
    xmin, xmax = 0.1, 0.99999
    dx = (xmax-xmin)/1000
    x_full_range = torch.range(xmin, xmax, dx)
    
    def get_ud(x, p):
        u = p[0]*torch.pow(x, p[1]) * torch.pow((1-x), p[2])
        d = p[3]*torch.pow(x, p[4]) * torch.pow((1-x), p[5])
        return u, d

    def integral_approx(y, dx):
        return torch.sum(y[:-1]) * dx

    def torch_interp(x, xs, ys):
        # determine the output data type
        ys = torch.Tensor(ys)
        dtype = ys.dtype

        # normalize data types
        ys = ys.type(torch.float64)
        xs = xs.type(torch.float64)
        x = x.type(torch.float64)

        # pad control points for extrapolation
        xs = torch.cat([torch.tensor([torch.finfo(xs.dtype).min]), xs, torch.tensor([torch.finfo(xs.dtype).max])], axis=0)
        ys = torch.cat([ys[:1], ys, ys[-1:]], axis=0)

        # compute slopes, pad at the edges to flatten
        ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
        ms = torch.nn.functional.pad(ms[:-1], (1, 1))

        # solve for intercepts
        bs = ys - ms*xs

        # search for the line parameters at each input data point
        # create a grid of the inputs and piece breakpoints for thresholding
        # rely on argmax stopping on the first true when there are duplicates,
        # which gives us an index into the parameter vectors
        
        #Argmax not implemented for boolean on CPU
        double_from_bool = xs[..., None, :] > x[..., None]
        double_from_bool = double_from_bool.double()
        i = torch.argmax(double_from_bool, dim=-1)
        m = ms[..., i]
        b = bs[..., i]

        # apply the linear mapping at each input data point
        y = m*x + b
        
        return torch.reshape(y, x.shape).type(dtype)




    def inverse_cdf(cdf_allx1):
        u = torch.rand((1*nevents,))*0.9999
        cdf_sort_indx1 = torch.argsort(cdf_allx1)
        cdf_sort1 = cdf_allx1[cdf_sort_indx1]
        x_sort1 = x_full_range[cdf_sort_indx1]
        
        events_out1 = torch_interp(u, cdf_sort1, x_sort1)
        events_out1 = torch.reshape(events_out1, (nevents,))
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
        
        inv_indices = torch.arange(pdf1.size(0)-1, -1, -1).long()
        inv_pdf1 = pdf1.index_select(0, inv_indices)
        
        inv_indices = torch.arange(pdf2.size(0)-1, -1, -1).long()
        inv_pdf2 = pdf2.index_select(0, inv_indices)
        
        inv_cdf_allx1 = (torch.cumsum(inv_pdf1*dx, dim=0)/torch.sum(pdf1*dx))
        inv_cdf_allx2 = (torch.cumsum(inv_pdf2*dx, dim=0)/torch.sum(pdf2*dx))
        
        indices = torch.arange(inv_cdf_allx1.size(0)-1, -1, -1).long()
        cdf_allx1 = inv_cdf_allx1.index_select(0, indices)
        
        indices = torch.arange(inv_cdf_allx2.size(0)-1, -1, -1).long()
        cdf_allx2 = inv_cdf_allx2.index_select(0, indices)
        
        events1 = inverse_cdf(cdf_allx1)
        events2 = inverse_cdf(cdf_allx2)

        events = torch.cat([torch.unsqueeze(events1, 0), torch.unsqueeze(events2, 0)], dim=0)

        return events, norm1, norm2
    
    events, norm1, norm2 = gen_events_sigmas(param)

    return events, norm1, norm2

def wrapperGenerator(params, parmin, parmax, nevents):
    
    return functorch.vmap(lambda x:paramsToEvents(x, parmin, parmax, nevents), in_dims=0, randomness="different")(params)