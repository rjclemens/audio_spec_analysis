from __future__ import print_function
from scipy import signal

import pyaudio
import wave
import matplotlib.pyplot as plot
import scipy.io.wavfile as spwav
import numpy as np


CHUNK = 1024 #number of frames per buffer
CHANNELS = 1 #monoaudio (stereo not available ??)
SAMPLE_RATE = 44100
FORMAT = pyaudio.paInt16 #16 bit signed integer
RECORD_SECONDS = 5
WAV_OUTPUT = "wav_output.wav"

p = pyaudio.PyAudio()

def write_stream():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(int((SAMPLE_RATE/CHUNK)*RECORD_SECONDS)):
        audio_data = stream.read(CHUNK) #read CHUNK bytes of stereoaudio stream
        frames.append(audio_data)

    stream.stop_stream()
    stream.close()
    p.terminate() #terminate the stream


    wf = wave.open(WAV_OUTPUT, mode='wb') #open file for writing
    wf.setnchannels(2)
    wf.setsampwidth(p.get_sample_size(FORMAT)) #num of bytes in paInt16
    wf.setframerate(SAMPLE_RATE) #44100 samples per second

    wf.writeframes(b''.join(frames)) #concatenate all the elements of frames into empty string b
    wf.close()

    print("* stopping")

def plot_wav(wav_file_name):
    wf_data = spwav.read(wav_file_name) #returns [sampling rate, data]
    sampling_rate = wf_data[0]
    audio_data = wf_data[1]

    plot.interactive(False)

    p1 = plot.figure(1)
    plot.plot(audio_data[0:(sampling_rate*RECORD_SECONDS)])
    plot.ylabel("Amplitude")
    plot.xlabel("Samples")


def fft(wav_file_name):
    fs_rate, sig = spwav.read(wav_file_name) # signal is np array
    shape = sig.shape
    print(sig)
    print("Shape:", shape)
    print("Frequency Sampling:", fs_rate)
    l_audio = len(shape) # num dimensions of array = num channels
    print("Channels:", l_audio)

    if l_audio == 2: # average out the two cols of array into one col (single channel)
        sig = sig.sum(axis=1)/2 # sum across columns: {[a,b],[c,d]}->{[a+b],[c+d]}

    N = shape[0] # num rows in signal np array
    secs = N / float(fs_rate) # total num samples
    ts = 1.0 / float(fs_rate) # sampling period
    t = np.arange(0, secs, ts) # time vector (numpy.ndarray)

    FFT = np.fft.fft(sig)[range((int) (N/2))] # only need half values bc even
    FFT_freqs = fftfreq(sig.size, t[1]-t[0])[range((int) (N/2))] # np.ndarray of possible frequencies

    p2 = plot.figure(2)
    plot.plot(t, sig, "g") # plotting the signal
    plot.xlabel('Time')
    plot.ylabel('Amplitude')

    p3 = plot.figure(3)
    plot.plot(FFT_freqs, FFT, "r") # plotting the complete fft spectrum
    plot.xlabel('Frequency (Hz)')
    plot.ylabel('Count dbl-sided')

    for i in range(5):
        max_index = np.argmax(FFT)
        max_freq = FFT_freqs[max_index]
        print("MAX VAL: ", abs(np.amax(FFT)), "MAX FREQ: ", max_freq)
        FFT[max_index] = 0 # reset to find next highest val


    
def spectrogram(wav_file_name):
    fs_rate, sig = spwav.read(wav_file_name)

    # convert stereo to mono
    if len(sig.shape) == 2: 
        sig = sig.sum(axis=1)/2 

    freqs, times, spectrogram = signal.spectrogram(sig, fs_rate)
    print(times)
    plot.pcolormesh(times, freqs, np.log(spectrogram))
    plot.ylabel('Frequency [Hz]')
    plot.xlabel('Time [msec]')



if __name__ == '__main__':
    write_stream()
    plot_wav(WAV_OUTPUT)
    spectrogram(WAV_OUTPUT)

    plot.show()

