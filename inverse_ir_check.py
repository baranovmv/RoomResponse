#!env python3
# Code in this file just checks how well we can inverse Room Impulse Response.

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fftpack
import math
import subprocess
import wavio
import sys
import sounddevice as sd
from optparse import OptionParser

from measure import Spectrum
from room_response_estimator import *

#######################################################################################################################

if __name__ == "__main__":

    # Parse console command options.
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

    # Just reuse previous measurements.
    if options.reuse_wav:
        ir_file = options.reuse_wav

    else:
        print("Reuse previous measurements in this module, invoke --reuse _file_ (-r)")


    # Get the result of measurement from wav file.
    ir_fl = wavio.read( ir_file )
    ir = ir_fl.data[0:,0]/math.pow(2.0, ir_fl.sampwidth*8-1)

    # Estimate Room Response from the raw signal.
    room_response = estimator.estimate(ir)

#######################################################################################################################
    # Estimating inverse impulse response.

    # Cut room response to some sensible length so that toeplitz matrix will form managable
    # system of linear equations:
    rr_len = 1024
    short_rr = room_response[:rr_len]

    # The length of th inverse room response that we're going to estimate:
    irr_len = 2048
    inv_room_response = estimator.inverted_ir(short_rr, irr_len)

    # Making the Y signal -- sum of couple of sines and awg-noise:
    time = np.array(range(irr_len*4))
    A = [0.5, 0.2]
    freq=[np.pi/16, np.pi/20]
    phi = [0, np.pi*1.3]
    Y_original = sum([a*np.sin(2*np.pi*f*time + p) for a,f,p in zip(A, freq, phi)])
    # Add some noise
    Y_original += np.random.normal(0, 0.27, size=Y_original.shape[0])

    # Prefilter it with inverse room response
    Y_predistorted = fftconvolve(Y_original, inv_room_response)[:Y_original.shape[0]]

    # Filter it like it was played through the speakers (note: we do it with the long version of
    # the room response):
    Y = fftconvolve(room_response, Y_predistorted)[:Y_predistorted.shape[0]]

    # Get rid of edge effect at the beginning:
    Y = Y[rr_len-1:]

    # The error:
    residuals = Y_original[:Y.shape[0]] - Y

    qdelta = fftconvolve(room_response, inv_room_response)

    plt.subplot(321)
    plt.plot( qdelta )
    plt.xlim([0, irr_len])
    plt.legend(["Quasi-delta impulse"])
    plt.grid()

    plt.subplot(322)
    plt.plot( *Spectrum(qdelta) )
    plt.xlim([options.lowfreq, options.highfreq])
    plt.legend(["Spectum of the quasi-delta impulse"])
    plt.grid()


    plt.subplot(323)
    plt.plot( Y_original )
    plt.plot( Y )
    plt.xlim([irr_len, irr_len+256])
    plt.legend(["Zoomed Y_original vs Played signal"])
    plt.grid()

    plt.subplot(324)
    plt.plot( *Spectrum(room_response[:irr_len]) )
    plt.plot( *Spectrum(inv_room_response) )
    plt.legend(["Spectrum of the Room Response and it's inversion"])
    plt.grid()


    plt.subplot(325)
    plt.plot( residuals )
    plt.legend(["Residual"])
    plt.grid()

    plt.show()
