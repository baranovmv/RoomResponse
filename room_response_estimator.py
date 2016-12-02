from scipy.fftpack import fft, ifft
from scipy.signal import fftconvolve, convolve
import numpy as np
import math

class RoomResponseEstimator(object):
    """
    Gives video impulse, gets response and calculate impulse response.
    Method from paper: "Simultaneous measurement of impulse response and distortion with a swept-sine technique"
    Angelo Farina
    """

    def __init__(self, Fs=44100):
        self.Fs = Fs

        # Total length in samples
        self.T = Fs*1
        self.w1 = 1.0 / self.Fs * 2*np.pi
        self.w2 = 20000.0 / self.Fs * 2*np.pi

        self.test_pulse = self.video()

        exp_window = 1-np.exp(np.linspace(0,-10, 5000))
        self.test_pulse[:exp_window.shape[0]] *= exp_window
        exp_window = 1-np.exp(np.linspace(0,-10, 1500))
        self.test_pulse[-exp_window.shape[0]:] *= exp_window[-1::-1]
        # self.test_pulse *= 1.0/np.sqrt(sum(np.square(self.test_pulse)))

        kend = 10**((-6*np.log2(self.w2/self.w1))/20)
        k = np.log(kend)/self.T

        self.reverse_pulse = self.test_pulse[-1::-1] * \
            np.array(list(\
                map(lambda t: np.exp(t*k), np.arange(self.T))\
                )\
            )

        Frp =  fft(fftconvolve(self.reverse_pulse, self.test_pulse))
        self.reverse_pulse /= np.abs(Frp[round(Frp.shape[0]/4)])

        # self.test_pulse = np.append(self.video(), np.zeros(4096))

    def video(self):

        w1 = self.w1
        w2 = self.w2
        T = self.T

        # page 5
        def lin_freq(t):
            return w1*t + (w2-w1)/T * t*t / 2

        # page 6
        def log_freq(t):
            K = T * w1 / np.log(w2/w1)
            L = T / np.log(w2/w1)
            return K * (np.exp(t/L)-1.0)

        freqs = log_freq(range(int(T)))
        impulse = np.sin(freqs)
        return impulse

    def estimate(self, response):

        # inverse = self.test_pulse[-1:0:-1]
        I = fftconvolve( response, self.reverse_pulse, mode='full')

        I = I[self.test_pulse.shape[0]:self.test_pulse.shape[0]*2+1]

        return I