import os
import soundfile


def scan_directory(dir_name):
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()

    addrs = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = subdir + file
                addrs.append(filepath)
    return addrs


def find_pair(noisy_file_name):
    clean_dirs = []
    for i in range(len(noisy_file_name)):
        addrs = noisy_file_name[i]
        if addrs.endswith(".wav"):
            clean_addrs = str(addrs).replace('noisy', 'clean')
            clean_dirs.append(clean_addrs)
    return clean_dirs


def find_srl_pair(noisy_file_name):
    srl_dirs = []
    for i in range(len(noisy_file_name)):
        addrs = noisy_file_name[i]
        if addrs.endswith(".wav"):
            srl_addrs = str(addrs).replace('noisy', 'srl_latents').replace('.wav', '.pth')
            srl_dirs.append(srl_addrs)
    return srl_dirs


def addr2wav(addr):
    wav, fs = soundfile.read(addr)
    return wav


# make a new dir
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

