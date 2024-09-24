import torch
from torch.utils.data import Dataset, DataLoader
from utils import scan_directory, find_pair, addr2wav, find_srl_pair
import random


def create_dataloader(opt, mode):
    if mode == 'train':
        dataset = Wave_Dataset_for_Stage2(opt, mode) if opt.stage2_training else Wave_Dataset(opt, mode)
        return DataLoader(
            dataset=dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        dataset = Wave_Dataset_for_Stage2(opt, mode) if opt.stage2_training else Wave_Dataset(opt, mode)
        return DataLoader(
            dataset=dataset,
            batch_size=opt.batch_size, shuffle=False, num_workers=0
        )
    elif mode == 'test':
        return DataLoader(
            dataset=Wave_Dataset_for_test(opt, mode),
            batch_size=1, shuffle=False, num_workers=0
        )


class Wave_Dataset(Dataset):
    def __init__(self, opt, mode):
        # load data
        self.mode = mode
        self.chunk_size = opt.chunk_size

        if mode == 'train':
            print('<Training dataset>')
            print('Load the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(opt.noisy_dirs_for_train)
            self.clean_dirs = find_pair(self.noisy_dirs)

        elif mode == 'valid':
            print('<Validation dataset>')
            print('Load the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(opt.noisy_dirs_for_valid)
            self.clean_dirs = find_pair(self.noisy_dirs)

    def __len__(self):
        return len(self.noisy_dirs)

    def __getitem__(self, idx):
        # read the wav
        inputs = addr2wav(self.noisy_dirs[idx])
        targets = addr2wav(self.clean_dirs[idx])

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        wav_len = len(inputs)
        assert wav_len == len(targets)

        # if wav_len < self.chunk_size:
        #     units = self.chunk_size // wav_len
        #     inputs_final = []
        #     targets_final = []
        #     for i in range(units):
        #         inputs_final.append(inputs)
        #         targets_final.append(targets)
        #     inputs_final.append(inputs[:self.chunk_size % wav_len])
        #     targets_final.append(targets[:self.chunk_size % wav_len])
        #     inputs = torch.cat(inputs_final, dim=-1)
        #     targets = torch.cat(targets_final, dim=-1)
        # # Randomly crop waveforms to the desired chunk size
        # else:
        #     stp = random.randint(0, len(inputs) - self.chunk_size)
        #     inputs = inputs[stp:stp + self.chunk_size]
        #     targets = targets[stp:stp + self.chunk_size]

        return inputs, targets


class Wave_Dataset_for_Stage2(Dataset):
    def __init__(self, opt, mode):
        # load data
        self.mode = mode
        self.chunk_size = opt.chunk_size

        if mode == 'train':
            print('<Training dataset>')
            print('Load the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(opt.noisy_dirs_for_train)
            self.clean_dirs = find_pair(self.noisy_dirs)
            self.srl_latent_dirs = find_srl_pair(self.noisy_dirs)

        elif mode == 'valid':
            print('<Validation dataset>')
            print('Load the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(opt.noisy_dirs_for_valid)
            self.clean_dirs = find_pair(self.noisy_dirs)
            self.srl_latent_dirs = find_srl_pair(self.noisy_dirs)

    def __len__(self):
        return len(self.noisy_dirs)

    def __getitem__(self, idx):
        # read the wav
        inputs = addr2wav(self.noisy_dirs[idx])
        targets = addr2wav(self.clean_dirs[idx])
        srl_latents = torch.load(self.srl_latent_dirs[idx])
        srl_latents = torch.cat([srl_latents[0], srl_latents[1]], dim=0)

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        wav_len = len(inputs)
        assert wav_len == len(targets)

        # if wav_len < self.chunk_size:
        #     units = self.chunk_size // wav_len
        #     inputs_final = []
        #     targets_final = []
        #     for i in range(units):
        #         inputs_final.append(inputs)
        #         targets_final.append(targets)
        #     inputs_final.append(inputs[:self.chunk_size % wav_len])
        #     targets_final.append(targets[:self.chunk_size % wav_len])
        #     inputs = torch.cat(inputs_final, dim=-1)
        #     targets = torch.cat(targets_final, dim=-1)
        # # Randomly crop waveforms to the desired chunk size
        # else:
        #     stp = random.randint(0, len(inputs) - self.chunk_size)
        #     inputs = inputs[stp:stp + self.chunk_size]
        #     targets = targets[stp:stp + self.chunk_size]

        return inputs, targets, srl_latents


class Wave_Dataset_for_test(Dataset):
    def __init__(self, opt, mode):
        # load data
        self.mode = mode

        if mode == 'test':
            print('<Test dataset>')
            print('Load the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(opt.test_database)
            self.clean_dirs = find_pair(self.noisy_dirs)
        else:
            raise Exception("Mode error!")

    def __len__(self):
        return len(self.noisy_dirs)

    def __getitem__(self, idx):
        # read the wav
        inputs = addr2wav(self.noisy_dirs[idx])
        targets = addr2wav(self.clean_dirs[idx])

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        wav_len = len(inputs)
        assert wav_len == len(targets)

        return inputs, targets, self.noisy_dirs[idx]
