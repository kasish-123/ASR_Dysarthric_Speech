import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import scipy.signal
import shutil
import pyprog
import utilities

SETTINGS_DIR = os.path.dirname(os.path.realpath('__file__'))

def create_directories(directories_list, base_path="images/Train/Control/"):
    for item in directories_list:
        if not os.path.exists(os.path.join(base_path, item)):
            os.makedirs(os.path.join(base_path, item))

def organise_waves_in_directories(waves_path, new_path="images/Train/Control/"):
    for directory, _, files in os.walk(waves_path):
        for f in files:
            if f.endswith(".wav"):
                file_path = os.path.join(directory, f)
                for label in dictionary.iloc[:, 1]:
                    if "_" + label + "_" in f:
                        print("Moving", f)
                        shutil.move(file_path, os.path.join(new_path, label, f))
                        break

def get_wav_info(wav_file):
    sample_rate, frames = wav.read(wav_file)
    sound_info = np.array(frames, dtype='int16')
    return sound_info, sample_rate

def save_a_spectrogram(wav_file_path, file_name):
    sound_info, frame_rate = get_wav_info(wav_file_path)
    nperseg = int(0.2 * frame_rate)  # 200 ms frames
    noverlap = int(0.12 * frame_rate)  # 80 ms overlap
    nfft = 256  # FFT size
    
    f, t, Sxx = scipy.signal.spectrogram(sound_info, fs=frame_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.axis('off')
    
    plt.savefig(os.path.splitext(wav_file_path)[0]+'.jpg', format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

def create_spectrograms(waves_path):
    inp = input("All wave files in "+waves_path+" will be deleted. Do you want to proceed? y/n")
    if inp.lower() == "y":
        prog = pyprog.ProgressBar("Creating spectrograms: ", " Done",
                                  utilities.get_no_files_in_path(waves_path))
        no_processed = 0
        prog.update()
        
        for directory, _, files in os.walk(waves_path):
            for f in files:
                if f.endswith(".wav"):
                    file_path = os.path.join(directory, f)
                    save_a_spectrogram(file_path, f)
                    os.remove(file_path)
                    no_processed += 1
                    prog.set_stat(no_processed)
                    prog.update()
        prog.end()

database_path = os.path.join(SETTINGS_DIR, "UASPEECH/Dysarthric/Test/M01/")
dictionary = pd.read_csv("dictionary_UASPEECH.csv")
base_path = os.path.join(SETTINGS_DIR, "images/Dysarthric/Test/M01/")

create_directories(dictionary.iloc[:,1], base_path=base_path)

for directory, _, _ in os.walk(base_path):
    if "_UW" in directory:
        shutil.rmtree(directory)
        print("Deleted", directory)

organise_waves_in_directories(database_path, new_path=base_path)

create_spectrograms(base_path)
