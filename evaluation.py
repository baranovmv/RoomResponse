#!env python3

import numpy as np
import matplotlib.pyplot as plt
import wavio
import scipy.signal
import scipy.fftpack
import math
import sys

import signal_env
from room_response_estimator import *

#######################################################################################################################

def Spectrum(s):
    Ftest = scipy.fftpack.fft( s )
    n = round(s.shape[0]/2)
    xf = np.linspace(0.0, 44100/2.0, n)
    return xf, 20*np.log10(np.abs(Ftest[0:n])) 

#######################################################################################################################

estimator = RoomResponseEstimator()

# Get impulse response of the room.
# This one is Cripta di San Sebastiano 
# "Ipogeum space carved into the rock. It's with the presence of the humidity,
# unconfined and influenced by weather conditions."
# http://www.openairlib.net/auralizationdb/content/cripta-di-san-sebastiano-sternatia
ir_fl = wavio.read( 'ir1_-_iringresso_new.wav' )
ir = ir_fl.data[:13400,0]/math.pow(2.0, ir_fl.sampwidth*8-1)

sig_env = signal_env.SoundEnvirenment(ir, ir_fl.rate)

in_sig = np.array([])
in_sig = np.append(in_sig, sig_env.run(np.zeros(1024)))
in_sig = np.append(in_sig, sig_env.run(estimator.probe_pulse))
in_sig = np.append(in_sig, sig_env.run(np.zeros(10240)))

room_response = estimator.estimate(in_sig)

#######################################################################################################################
# Display

delta = scipy.signal.fftconvolve( estimator.reverse_pulse, estimator.probe_pulse)
# room_response = room_response[:ir.shape[0]]
ir = ir[:13400]
offset = 1223
error = ir-room_response[offset:13400+offset]

plt.subplot(311)
legends = []
legend, = plt.plot(ir, 'b', label="True IR")
legends.append(legend)
legend, = plt.plot(room_response[offset:], 'r', label="Estimated IR" )
legends.append(legend)
plt.axis([-10,500, -1, 1])
# plt.axis([0,2000,-1,1])
plt.grid()
plt.legend(handles=legends)

plt.subplot(312)
legends = []
legend, = plt.plot(error, label="Absolute error")
legends.append(legend)
plt.grid()
plt.legend(handles=legends)

plt.subplot(313)
legends = []
xf,R = Spectrum(ir)
legend, = plt.plot( xf, R, label="Spectrum of the true IR" )
legends.append(legend)

exf, eR = Spectrum(room_response)
legend, = plt.plot( exf, eR, label="Spectrum of the estimated IR" )
legends.append(legend)
plt.legend(handles=legends)
plt.grid()

plt.show()
