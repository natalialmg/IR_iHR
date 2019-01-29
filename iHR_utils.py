from scipy.signal import butter, firwin
import numpy as np
import sq_stft_utils as sq
import scipy
import matplotlib.pyplot as plt
from biosppy.signals import ecg, bvp
# Some Functions
def butter_bandpass(lowcut, highcut,fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a 

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def butter_highpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="highpass")
    return b, a

def optimal_svd(Y):
    ### OPTIMAL SHRINKAGE SVD
    # Implementation of algorithm proposed in:
    # based on the following paper: Gavish, Matan, and David L. Donoho. "Optimal shrinkage of singular values." IEEE Transactions on Information Theory 63.4 (2017): 2137-2152.
    #
    U,s,V = np.linalg.svd(Y, full_matrices=False)
    m,n =  Y.shape
    beta = m/n
    
    y_med = np.median(s)


    beta_m = (1- np.sqrt(beta))**2
    beta_p = (1+ np.sqrt(beta))**2

    t_array = np.linspace(beta_m,beta_p,100000)
    dt = np.diff(t_array)[0]

    f = lambda t: np.sqrt((beta_p-t)*(t-beta_m))/(2*np.pi*t*beta)
    F = lambda t: np.cumsum(f(t)*dt)


    mu_beta =  t_array[np.argmin((F(t_array)-0.5)**2)]


    sigma_hat = y_med/np.sqrt(n*mu_beta)

    def eta(y,beta):
        mask =(y>=(1+np.sqrt(beta)))
        aux_sqrt = np.sqrt((y[mask>0]**2-beta-1)**2 - 4*beta)
        aux = np.zeros(y.shape)
        aux[mask>0] = aux_sqrt/y[mask>0]
        return mask*aux

    def eta_sigma(y,beta,sigma):
        return sigma*eta(y/sigma,beta)

    s_eta = eta_sigma(s,beta,sigma_hat*np.sqrt(n))
    
#     trim U,V
    aux=s_eta>0
    
    return U[:,aux], s_eta[aux], V[aux,:]

def curveExtractor(E, lamb, guess=None):

    eps = 1e-8
    E = np.abs(np.array(E)).T
    E /=np.sum(E)
    E = np.log(E+eps)

    m,n = E.shape # m is time points, n is frequency

    curve = np.zeros([m])
    if guess is  None:
        curve[0] = np.argmax(E[0,:])
    else:
        curve[0] = guess
#     print(curve[0])

    freq_indexes = np.arange(n)
    for i in np.arange(m)[1:]:
        penalty = (curve[i-1]-freq_indexes)**2
        full_term = E[i,:] - lamb * penalty
        curve[i] = np.argmax(full_term)

    return curve

def get_iHR_ecg(signal_ecg,sampling_rate,ts = 0):
    #ts is synchronization delay (e.g., STFT window delay)
    
    out = ecg.ecg(signal=signal_ecg, sampling_rate=sampling_rate, show=False)
    
    time_I = out['heart_rate_ts']
    curve_I = out['heart_rate']
    
    curve_I = curve_I[time_I>ts+time_I[0]]
    time_I = time_I[0:curve_I.shape[0]]-time_I[0]

#     plt.plot(time_eeg,curve_eeg)
#     plt.show()
    
    return curve_I, time_I


def quality_process(signals,prior_bpm,fs,window = 301,fL=0.4,fH = 5, fp_d = 0.7,fp_u = 5,verbose = False):
    nv = signals.shape[0]
    quality_array = np.zeros([nv])
    for i in np.arange(nv):
        senal = signals[i,:]

        b,a = butter_bandpass(fL, fH,fs, order=5)
        filtered_signal = scipy.signal.filtfilt(b, a, senal)
        t_v = np.arange(filtered_signal.shape[0])/fs

        stft_v, f_v,_,_= sq.SST_helper(filtered_signal, fs, fH/fs, fL/fs, windowLength=window)

        f_v = f_v*fs
        nu = np.argmin(np.abs(f_v-fp_u))
        nd = np.argmin(np.abs(f_v-fp_d))
        stft_vp = np.abs(stft_v[nd:nu,:])
        stft_vp = stft_vp[::-1,:]
        
        ## Quality index ##
        prior_f =prior_bpm/60
        f_low_prior = prior_f*0.75
        f_high_prior = prior_f*1.25
        idx_prior = np.logical_and(f_v>f_low_prior,f_v<f_high_prior)
        idx_band = np.logical_and(f_v>0.25*prior_f,f_v<2*prior_f)

        power= np.abs(stft_v).sum(1)
        power_fraction = power[idx_prior].sum()/power[idx_band].sum()
        quality_array[i] = power_fraction
        
        
        print('Signal :: ' + str(i))
        print('Power fraction', power_fraction)
        if verbose:
            
            #Plot Signals
            plt.figure(figsize = (12,5))
            plt.subplot(2,1,1)
            plt.plot(t_v,filtered_signal)
            plt.xlim([t_v[0],t_v[-1]])
            plt.subplot(2,1,2)
            plt.imshow(stft_vp,extent=[t_v[0], t_v[-1],f_v[nd]*60,f_v[nu]*60],
                       aspect = 'auto',cmap='Blues')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [bpm]')
            plt.show()

    return quality_array



def compute_RMSE_every_n(signal1,signal2, sampling_time,mean_time):
    signal1_m,time_m = strided_mean(signal1,sampling_time, mean_time)
    signal2_m,_ = strided_mean(signal2,sampling_time, mean_time)
    
    RMSE = np.sqrt(np.mean((signal1_m- signal2_m)**2))
    rRMSE = np.mean(np.abs(signal1_m- signal2_m)/signal1_m)
    return RMSE, rRMSE


def strided_mean(signal,sampling_time, mean_time):
   block_len = np.ceil(mean_time/sampling_time)
   n_blocks = np.floor(signal.shape[0]/block_len)
   mean_sig = np.zeros([int(n_blocks),1])
   mean_t =np.arange(n_blocks)*mean_time
   for i in np.arange(n_blocks):
       mean_sig[int(i)]= signal[int(i*block_len):int((i+1)*block_len),...].mean()
   return mean_sig, mean_t


