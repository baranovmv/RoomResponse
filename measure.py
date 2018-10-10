#!env python3


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fftpack
import math
import subprocess
import wavio
import scipy.io.wavfile
import sys
import sounddevice as sd
from optparse import OptionParser

import signal_env
from room_response_estimator import *

#######################################################################################################################

def Spectrum(s):
    Ftest = scipy.fftpack.fft( s )
    n = round(s.shape[0]/2)
    xf = np.linspace(0.0, 44100/2.0, n)
    return xf, 20*np.log10(np.abs(Ftest[0:n]))

#######################################################################################################################
if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option( "-r", "--reuse", action="store",
                        type="string", dest="reuse_wav",
                        help="Use wav file with previous record instead actual playing and recording.")
    parser.add_option( "-d", "--duration", action="store",
                        type="float", dest="duration",
                        default=10,
                        help="Duration of probe impulse.")
    parser.add_option( "-b", "--low-freq", action="store",
                        type="float", dest="lowfreq",
                        default=100,
                        help="The lowest detected frequency [Hz].")
    parser.add_option( "-e", "--high-freq", action="store",
                        type="float", dest="highfreq",
                        default=15000,
                        help="The highest frequency in probe impulse [Hz].")
    (options, args) = parser.parse_args()


    estimator = RoomResponseEstimator(options.duration, options.lowfreq, options.highfreq)

    if options.reuse_wav:
        ir_file = options.reuse_wav
    else:
        # Store probe signal in wav file.
        x = np.append( np.zeros(44100), estimator.probe_pulse * 0.1 )
        scipy.io.wavfile.write("test_sweep.wav", 44100, x)

        # Play it and record microphones input simultaneously.
        reccommand = \
            "rec -q --clobber -r 44100 -b 16 -D -c 2 record_sweep.wav trim 0 10".split(" ")
        prec = subprocess.Popen(reccommand)

        playcommand = \
            "play -q test_sweep.wav".split(" ")
        pplay = subprocess.Popen(playcommand)
        pplay.wait()
        prec.wait()

        ir_file = "record_sweep.wav"


    # Get the result of measurement from wav file.
    ir_fl = wavio.read( ir_file )
    ir = ir_fl.data[0:,0]/math.pow(2.0, ir_fl.sampwidth*8-1)

    # Restore Room Response from the raw signal.
    room_response = estimator.estimate(ir)

    # Derive inverted room response for active room correction.
    Hi = fft(room_response)
    lmbd = 1e-2
    # Perform Weiner deconvolution.
    inv_room_response = np.real(ifft(np.conj(Hi)/(Hi*np.conj(Hi) + lmbd**2)))
    inv_room_response /= np.max(np.abs(inv_room_response))

    deconvolved_ir = fftconvolve(room_response, inv_room_response)
    deconvolved_sweep = fftconvolve(ir, inv_room_response)
#######################################################################################################################

    plt.subplot(321)
    plt.plot(room_response)
    plt.legend(["Measured Room Response"])
    x0 = np.argmax(room_response)
    plt.xlim([x0, x0+1024])
    plt.grid()

    plt.subplot(322)
    plt.plot(deconvolved_ir)
    plt.legend(["Deconvolved Room Response"])
    x0 = np.argmax(deconvolved_ir)
    plt.xlim([x0-100, x0+100])
    ymin, ymax = plt.ylim()
    plt.ylim( [ymin, ymax*1.6])
    plt.grid()

    plt.subplot(323)
    plt.plot(ir)
    plt.legend(["Recorded log-sine sweep"])
    plt.grid()

    ax = plt.subplot(324)
    plt.plot(*Spectrum(ir))
    plt.legend(["Spectrum of the record"])
    ax.set_xscale('log')
    ymin, ymax = plt.ylim()
    plt.ylim( [ymin, ymax*1.6])
    plt.xlim([10, 2e4])
    plt.grid()

    plt.subplot(325)
    plt.plot(deconvolved_sweep)
    plt.legend(["Deconvolved sine sweep"])
    plt.grid()

    ax = plt.subplot(326)
    plt.plot( *Spectrum(deconvolved_sweep))
    plt.legend(["Spectrum of deconvolved sine sweep" ])
    ax.set_xscale('log')
    ymin, ymax = plt.ylim()
    plt.ylim( [ymin, ymax*1.6])
    plt.xlim([10, 2e4])
    plt.grid()

    plt.show()
