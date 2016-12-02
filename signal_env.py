import scipy.signal
import numpy as np
import math

class SoundEnvirenment:
    """
    Emulates the envirenment. Method run(...)
    gets samples to render with loadspeaker and returns measured
    sample from a microphone.
    """

    def __init__(self, ir, Fs=44100):
        self.Fs = Fs
        # self.Hspeaker = np.append( np.zeros(100), [1] )
        self.Hspeaker = np.append( [1], np.zeros(100) )
        self.speaker_conv_tail = np.zeros( self.Hspeaker.shape[0] - 1 )
        self.Hroom = np.array(ir)
        self.Hroom = np.append( np.zeros(200), self.Hroom )
        self.room_conv_tail = np.zeros( self.Hroom.shape[0] - 1 )
        self.noiser = self.noise_src(Fs)
        self.t = 0

    def run(self, samples_in):
        # How many samples we should pass through
        Nin = samples_in.shape[0]

        out_2_input, self.speaker_conv_tail = self.conv(samples_in, self.Hspeaker, self.speaker_conv_tail)

        # Generate a noise we're hoping to eliminate.
        next_t = self.t+Nin/self.Fs
        noise = np.array([self.noiser(float(i)/self.Fs+self.t) for i in range(Nin)])
        self.t = next_t

        res, self.room_conv_tail = self.conv( noise+out_2_input, self.Hroom, self.room_conv_tail )
        # return out_2_input
        return res

    def conv(self, x, IR, tail):
        """
        Here we convolve output samples with loudspeaker IR and manage with tail,
        which left after convolution.
        """
        assert( IR.shape[0] == tail.shape[0]+1 )
        
        # Use time-domain convolution if batches are small.
        res = scipy.signal.fftconvolve(x, IR, mode='full')
        res[:tail.shape[0]] += tail
        tail = res[-tail.shape[0]:]
        # Cut the tail.
        return res[:x.shape[0]], tail

    def noise_src(self, sampling_freq=8000):
        # Random phase
        rand_phi = np.random.rand()*2*np.pi
        # Sin frequency.
        rand_F = (np.random.rand()*np.pi/64 + np.pi/4) / 2 / np.pi * self.Fs
        # F = np.pi/16 * Fs

        A = [0.5, 0.2]
        freq=[rand_F, rand_F*1.7]
        phi = [np.random.rand()*2*np.pi, np.random.rand()*2*np.pi]
        SNR = 40 # dB
        signal_power = np.square(np.linalg.norm(A))
        # SNR = Psig/Pnoise
        # Sigma_noise = Psig/SNR
        noise_sigma = np.sqrt(signal_power / math.pow( 10, SNR/20 ))
        def awgn(time):
            return np.random.normal(0, noise_sigma)

        def fan(time):
            f = sum([a*np.sin(2*np.pi*f*time + p) for a,f,p in zip(A, freq, phi)])
            f = f + awgn(time)
            return f

        def single_harm(time):
            np.sin(rand_F*time + rand_phi)*0.5
        def zero(time):
            return 0
        return awgn
