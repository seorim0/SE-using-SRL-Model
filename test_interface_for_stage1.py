"""
Test interface for speech enhancement!
You can just run this file.
"""
import argparse
import torch
import options
import utils
import random
import numpy as np
import time
from dataloader import create_dataloader
from scipy.io.wavfile import write
from s3prl.nn import S3PRLUpstream, Featurizer

######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
opt.arch = 'Stage1'
print(opt)

######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# set device
DEVICE = torch.device(opt.device)
# set seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
# define model
model = utils.get_arch(opt)
total_params = utils.cal_total_params(model)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))
# load the params
print('Load the pretrained model...')
chkpt = torch.load(opt.pretrain_model_path, map_location='cuda:0')
model.load_state_dict(chkpt['model'])
model = model.to(DEVICE)

srl_model = S3PRLUpstream("wavlm_large").to(DEVICE)
featurizer = Featurizer(srl_model)
featurizer = featurizer.to(DEVICE)
featurizer.load_state_dict(chkpt['featurizer'])

######################################################################################################################
######################################################################################################################
#                                             Main program - train                                                   #
######################################################################################################################
######################################################################################################################
print('Test start...')
st_time = time.time()
opt.test_database = opt.noisy_dirs_for_test
test_loader = create_dataloader(opt, mode='test')
data_num = 0
cln_all = []
enh_all = []

# test
model.eval()
srl_model.eval()
featurizer.eval()

if opt.wav_write_flag:
    utils.mkdir('./results/{}'.format(opt.test_name))

with torch.no_grad():
    for inputs, targets, dir_name in utils.Bar(test_loader):
        data_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        input_specs = model.cstft(inputs)
        input_real, input_imag = utils.power_compress(input_specs, cut_len=opt.fft_len // 2 + 1)
        out_real, out_imag, _, srl_guided_r, srl_guided_i = model.test_interface(input_real, input_imag)

        srl_guided_r, srl_guided_i = srl_guided_r[0].cpu().detach(), srl_guided_i[0].cpu().detach()

        save_addr = str(dir_name[0]).replace('noisy', 'srl_latents').replace('.wav', '.pth')
        torch.save([srl_guided_r, srl_guided_i], save_addr)

        out_specs = utils.power_uncompress(out_real, out_imag)

        outputs = model.cistft(out_specs)
        outputs = outputs.squeeze(1)

        # get score
        noisy_wavs = inputs.cpu().detach().numpy()[:, :outputs.size(1)]
        clean_wavs = targets.cpu().detach().numpy()[:, :outputs.size(1)]
        enhanced_wavs = outputs.cpu().detach().numpy()

        if opt.wav_write_flag:
            write('./results/{}/{}_{}_enhanced.wav'.format(opt.test_name, data_num, opt.test_name), opt.fs, enhanced_wavs[0])

        cln_all.extend(clean_wavs)
        enh_all.extend(enhanced_wavs)
        del inputs, targets, outputs, input_specs, input_real, input_imag, out_real, out_imag
        torch.cuda.empty_cache()

        cln_all.extend(clean_wavs)
        enh_all.extend(enhanced_wavs)

        del inputs, targets, outputs, input_specs, input_real, input_imag, out_real, out_imag, srl_guided_r, srl_guided_i
        torch.cuda.empty_cache()


test_log_fp = open('extra_code/test_log.txt', 'a')

avg_stoi = utils.cal_stoi_batch(cln_all, enh_all)
avg_csig, avg_cbak, avg_covl, avg_pesq, avg_ssnr = utils.cal_pysepm_metrics(cln_all, enh_all)
print('\nTotal score')
print('PESQ: {:.4f}  STOI: {:.4f}  CSIG {:.4f}  CBAK {:.4f}  COVL {:.4f}  SSNR {:.4f}'
      .format(avg_pesq, avg_stoi, avg_csig, avg_cbak, avg_covl, avg_ssnr))
print('System has been finished.')

test_log_fp.write(opt.pretrain_model_path + '\n\n')
test_log_fp.write('Total score\n')
test_log_fp.write('PESQ: {:.4f}  STOI: {:.4f}  CSIG {:.4f}  CBAK {:.4f}  COVL {:.4f}  SSNR {:.4f}\n'
                  .format(avg_pesq, avg_stoi, avg_csig, avg_cbak, avg_covl, avg_ssnr))

test_log_fp.close()