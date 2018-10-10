#!env python3
# The file creates wav-file with a probe impulse to be played through speakers.

import numpy as np
import scipy.signal
import math
import scipy.io.wavfile
import sys
from optparse import OptionParser

from room_response_estimator import *

#######################################################################################################################

def create_probe(RIR_estimator, filename):
    '''
    Generates probe signal, store it to a wav-file, and return its waveform.
    '''
    estimator = RoomResponseEstimator(options.duration, options.lowfreq, options.highfreq)
    # Store probe signal in wav file.
    x = np.append( np.zeros(round((RIR_estimator.Fs))), estimator.probe_pulse * 0.1 )
    scipy.io.wavfile.write(filename, round(RIR_estimator.Fs), x)

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
    parser.add_option( "-p", "--probe-file", action="store",
                        type="string", dest="probe_filename",
                        default="test_sweep.wav",
                        help="The probe sweep wave file name.")
    (options, args) = parser.parse_args()

    print("Storing probe signale to \"{}\"\nFrequency range [{}-{}], duration {}s".format(\
        options.probe_filename, options.lowfreq, options.highfreq, options.duration))
    estimator = RoomResponseEstimator(options.duration, options.lowfreq, options.highfreq)
    create_probe(estimator, options.probe_filename)
