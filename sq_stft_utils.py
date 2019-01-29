import numpy as np
import scipy
import scipy.signal
import scipy.special
import matplotlib.pyplot as plt
import pywt

def SST_STFT(x, lowFreq, highFreq, alpha, h=None, Dh=None, tDS=1, Smooth=True, Hemi=True):
    # function [tfr, tfrtic, tfrsq, tfrsqtic] = sqSTFTbase(x, lowFreq, highFreq, alpha, tDS, h, Dh, Smooth, Hemi)
    #
    # Python adaptation of Synchrosqueezing original matlab code: tfrrsp.m, by Hau-tieng Wu, 2013 
    # https://hautiengwu.wordpress.com/code/
    #
    # Paper: Daubechies, Ingrid, Jianfeng Lu, and Hau-Tieng Wu. "Synchrosqueezed wavelet transforms: An empirical mode decomposition-like tool." Applied and computational harmonic analysis 30.2 (2011): 243-261.
    # Thakur, Gaurav, et al. "The synchrosqueezing algorithm for time-varying spectral analysis: Robustness properties and new paleoclimate applications." Signal Processing 93.5 (2013): 1079-1094.

    xrow = x.size
    x = np.squeeze(x)

    # print(x.shape, x.T)

    t = np.arange(x.size)
    tLen = t[::tDS].size
    
        # for tfr
    # N = length([-0.5+alpha:alpha:0.5]) 
    N = np.arange(-0.5+alpha,0.5, alpha).size +1

    Nrange= int(N/2)
    
        # for tfrsq
    #Lidx = round( (N/2)*(lowFreq/0.5) ) + 1
    Lidx = np.round((N / 2) * (lowFreq / 0.5)) #TODO
    # Hidx = round( (N/2)*(highFreq/0.5) )
    Hidx = np.round((N / 2) * (highFreq / 0.5))-1 #TODO

    fLen = int(Hidx - Lidx + 1)
    
    
    
    #====================================================================
    ## check input signals
    if highFreq > 0.5:
        print('TopFreq must be a value in [0, 0.5]')
        return
    elif (tDS < 1) or (np.remainder(tDS,1)!=0):
        print('tDS must be an integer value >= 1')
        return

    h = h.T
    [hrow, hcol] = h.shape
    h = np.squeeze(h)
    Dh = Dh.T
    Dh = np.squeeze(Dh)

    # [hrow,hcol] = size(h); Lh = (hrow-1)/2
    Lh = (hrow-1)/2
    # print(hrow,hcol,Lh)
    if (hcol!=1)or (np.remainder(hrow,2)==0):
        print('H must be a smoothing window with odd length')
        return
    

    #====================================================================
        ## run STFT and reassignment rule
    tfrtic = np.linspace(0, 0.5, N/2)
    tfrsqtic = np.linspace(lowFreq, highFreq, fLen)
    tfrsq = np.zeros([tfrsqtic.size, tLen], dtype='complex')
    tfr = np.zeros([tfrtic.size, tLen], dtype='complex')  # for h

    # print("TFRs")
    # print(tfr.shape,tfrtic.shape,tfrsq.shape, tfrsqtic.shape)

    
    #Ex = mean(abs(x(min(t):max(t))).^2)
    Ex = np.mean(np.abs(x) ** 2)
    Threshold = 1.0e-8*Ex  # originally it was 1e-6*Ex
    
    Mid = np.round(tfrsqtic.size/2).astype('int')
    Delta = 20*(tfrsqtic[2]-tfrsqtic[1])**2
    weight = np.exp(-(tfrsqtic[Mid-10:Mid+10]-tfrsqtic[Mid]) **2/Delta)
    weight = weight / sum(weight)
    # weightIDX = [Mid-10:Mid+10] - Mid
    weightIDX = np.arange(Mid-10,Mid+10+1)-Mid

    # print("extras")
    # print(Mid,Delta,weight,weightIDX)

    for tidx in np.arange(tLen):
    
        ti = t[(tidx-1)*tDS+1]+1
        # tau = -min([round(N/2)-1,Lh,ti-1]):min([round(N/2)-1,Lh,xrow-ti])
        A = -np.min(np.array([np.round(N/2)-1,Lh,ti-1]))
        B = np.min(np.array([np.round(N/2)-1, Lh, xrow-ti]))+1
        ti=ti-1
        tau = np.arange(A,B)
        tau = tau.astype('int')
        indices= np.remainder(N+tau,N)

        LhTau = (Lh+tau).astype('int')
        norm_h = np.linalg.norm(h[LhTau])

        # norm_h=norm(h(Lh+1+tau))
        #norm_h = h(Lh+1)

        # print("indices", indices)

    
        tf0 = np.zeros(N)
        tf1 = np.zeros(N)
        tf0[indices] = x[ti+tau]* np.conj(h[LhTau])/norm_h #TODO! revisar el -1
        tf1[indices] = x[ti+tau]* np.conj(Dh[LhTau])/norm_h #TODO! revisar el -1

        # print("tf0", tf0)
        # print("ti",ti)
        # print("ti+tau",ti+tau)
        # print("LhTau",LhTau)
        # print("h[LhTau]",h[LhTau].T)
        # print("x[ti+tau]",x[ti+tau].T)
        # print("tf0[indices]", tf0[indices])
        # print("Dh",Dh)
        # print("Dh", Dh[144:176])
        # print("Dh[LhTau]", Dh[LhTau].T)
        # print("tf1[indices]", tf1[indices])

        # tf0(indices) = x(ti+tau).*conj( h(Lh+1+tau)) /norm_h
        # tf1(indices) = x(ti+tau).*conj(Dh(Lh+1+tau)) /norm_h
        tf0 = scipy.fftpack.fft(tf0)[np.arange(Nrange)]
        tf1 = scipy.fftpack.fft(tf1)[np.arange(Nrange)]

        # print("tf0", tf0.shape)
        # print(tf0)
        # print("tf1", tf1.shape)
        # print(tf1)

        # tf0 = fft(tf0) ; tf0 = tf0(1:N/2)
        # tf1 = fft(tf1) ; tf1 = tf1(1:N/2)
    
            # get the first order omega
        omega = np.zeros(tf1.size)


        avoid_warn = np.where(tf0!=0)
        # print(avoid_warn)
        omega[avoid_warn] = np.round(np.imag(N*tf1[avoid_warn]/tf0[avoid_warn]/(2.0*np.pi)))
        # print("omega", omega.shape)
        # print(omega[avoid_warn])
    
    #if tidx > 100 ; keyboard ; end
        sst = np.zeros(fLen, dtype='complex')
    
        for jcol in range(np.round(N/2).astype('int')):
            if np.abs(tfr[jcol,0]) > Threshold:
                jcolhat = jcol - omega[jcol]

                # print("above thr")
                # print(np.abs(tfr[jcol, 0]))
                # print(omega.shape)
                # print(jcolhat)
                #jcolhat = rem(rem(jcolhat-1,N)+N,N)+1

                #if (jcolhat < Hidx + 1) & (jcolhat >= Lidx)
                #	sst(jcolhat-Lidx+1) = sst(jcolhat-Lidx+1) + tf0(jcol) 
                #end
    
                if (jcolhat <= Hidx) & (jcolhat >= Lidx):
                    #IDXa = unique(min(Hidx, max(Lidx, jcolhat-Lidx+1+weightIDX))) 
    
                    if Smooth:
                        IDXb = np.where((jcolhat-Lidx+weightIDX <= Hidx) & (jcolhat-Lidx+weightIDX >= Lidx))
                        IDXa = jcolhat-Lidx+weightIDX[IDXb]
    
    
                        if Hemi:
                            if np.real(tf0(jcol)) > 0:
                                sst[IDXa] = sst[IDXa] + tf0[jcol]*weight[IDXb]
                            else:
                                sst[IDXa] = sst[IDXa] - tf0[jcol]*weight[IDXb]
                        else:
                            sst[IDXa] = sst[IDXa] + tf0[jcol]*weight[IDXb]
    
                    else:
    
                        if Hemi:
                            if (np.real(tf0[jcol]) > 0):
                                sst[int(jcolhat-Lidx)] = sst[int(jcolhat-Lidx)] + tf0[jcol]
                            else:
                                sst[int(jcolhat-Lidx)] = sst[int(jcolhat-Lidx+1)] - tf0[jcol]
                        else:
                            sst[int(jcolhat-Lidx)] = sst[int(jcolhat-Lidx)] + tf0[jcol]
        tfr[:, tidx] = tf0
        tfrsq[:, tidx] = sst
    
    return tfr, tfrtic, tfrsq, tfrsqtic

def hermf(N,M,tm):
    # computes a set  of orthonormal Hermite functions #
    # input: - N: number of points(must be odd)
    # - M: maximum order
    # - tm: half time support( >= 6 recommended)
    #
    # output: - h: Hermite functions(MxN)
    # - Dh: H ' (MxN)
    # - tt: time vector(1 xN)

    dt = 2 * tm / (N - 1)
    tt = np.linspace(-tm, tm, N)
    g = np.exp(-tt **2 /2)

    P = np.zeros([M+1,N])
    Htemp = np.zeros([M+1,N])
    Dh=np.zeros([M,N])

    P[0,:] = 1
    P[1,:] = 2*tt
    for k in range(2,M):
        P[k,:] = 2 * tt * P[k - 1,:] - 2 * (k - 2) * P[k - 2,:]

    for k in range(M+1):
        Htemp[k,:]= P[k,:] * g / np.sqrt(np.sqrt(np.pi) * 2 ** (k +1- 1) * scipy.special.gamma(k+1)) * np.sqrt(dt)
        # print("k",k)
    # print(Htemp.shape, Htemp)
    h = Htemp[0:M,:]

    # if (M > 1):
    for k in range(M):
        # print(k)
        Dh[k,:] =(tt * Htemp[k,:] -np.sqrt(2*(k+1)) * Htemp[k+1,:]) * dt
    # print(Dh.shape)

    return h, Dh


def SST_helper(x, Hz, highFreq, lowFreq, windowLength=377):
    FrequencyAxisResolution = 0.001
    h, Dh = hermf(windowLength, 1, 2)
    tfr, tfrtic, tfrsq, tfrsqtic = SST_STFT(x, lowFreq, highFreq, FrequencyAxisResolution, h, Dh, 1,
                                                           False, False)

    return tfr, tfrtic, tfrsq, tfrsqtic