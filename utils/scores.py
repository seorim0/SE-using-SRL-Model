import torch
from pesq import pesq
from pystoi import stoi
from scipy.linalg import toeplitz
from joblib import Parallel, delayed
import numpy as np
import pysepm


def cal_pesq(clean_wav, dirty_wav, FS=16000):
    try:
        pesq_score = pesq(FS, clean_wav, dirty_wav, "wb")
    except:
        print(' No utterances error')
        pesq_score = -1
    return pesq_score


def cal_pesq_batch(clean_wavs, dirty_wavs, FS=16000):
    pesq_score = Parallel(n_jobs=-1)(delayed(cal_pesq)(c, n, FS=FS) for c, n in zip(clean_wavs, dirty_wavs))
    pesq_score = np.array(pesq_score)
    return np.mean(pesq_score)


def cal_pesq_for_disc(clean_wavs, dirty_wavs, DEVICE='cuda'):
    pesq_score = Parallel(n_jobs=-1)(delayed(cal_pesq)(c, n) for c, n in zip(clean_wavs, dirty_wavs))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to(DEVICE)


def cal_stoi_batch(clean_wavs, dirty_wavs, FS=16000):
    stoi_score = Parallel(n_jobs=-1)(delayed(stoi)(c, n, FS, extended=False) for c, n in zip(clean_wavs, dirty_wavs))
    stoi_score = np.array(stoi_score)
    return np.mean(stoi_score)


def cal_pysepm_metrics(clean_wavs, enhanced_wavs):
    avg_csig_score = 0
    avg_cbak_score = 0
    avg_covl_score = 0
    avg_pesq_score = 0
    avg_ssnr_score = 0
    for i in range(len(enhanced_wavs)):
        csig_score, cbak_score, covl_score = pysepm.composite(clean_wavs[i], enhanced_wavs[i], fs=16000)
        ssnr_score = pysepm.SNRseg(clean_wavs[i], enhanced_wavs[i], fs=16000)
        pesq_score = pesq(16000, clean_wavs[i], enhanced_wavs[i], "wb")

        avg_csig_score += csig_score
        avg_cbak_score += cbak_score
        avg_covl_score += covl_score
        avg_pesq_score += pesq_score
        avg_ssnr_score += ssnr_score
    avg_csig_score /= len(enhanced_wavs)
    avg_cbak_score /= len(enhanced_wavs)
    avg_covl_score /= len(enhanced_wavs)
    avg_pesq_score /= len(enhanced_wavs)
    avg_ssnr_score /= len(enhanced_wavs)
    return avg_csig_score, avg_cbak_score, avg_covl_score, avg_pesq_score, avg_ssnr_score

