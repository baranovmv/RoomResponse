#!env python3
# The file acquire Impulse Response signal from response wav-file, stores it to another wav-file and display
# its plot and Spectrum.

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

from room_response_estimator import *
from measure import Spectrum

#######################################################################################################################

if __name__ == "__main__":
    parser = OptionParser()
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
    parser.add_option( "-r", "--response-file", action="store",
                        type="string", dest="response_wav",
                        default="response.wav",
                        help="The wav-file with recorded response.")
    (options, args) = parser.parse_args()

    print("Analyzing response file \"{}\"".format({options.response_wav}))
    estimator = RoomResponseEstimator(options.duration, options.lowfreq, options.highfreq)
    room_response_fl = scipy.io.wavfile.read( options.response_wav )
    response = room_response_fl[1]

    ir = estimator.estimate(np.array(response))

#######################################################################################################################
# Visualization

    plt.subplot(211)
    plt.plot(ir)
    plt.legend(["IR"])
    x0 = np.argmax(ir)
    plt.xlim([x0, x0+1024])
    plt.grid()

    ax = plt.subplot(212)
    plt.plot(*Spectrum(ir))
    plt.legend(["Spectrum of the IR"])
    # ax.set_xscale('log')
    ymin, ymax = plt.ylim()
    plt.xlim([options.lowfreq, options.highfreq])
    plt.ylim( [ymin-20, ymax+20])
    plt.grid()

    plt.show()
