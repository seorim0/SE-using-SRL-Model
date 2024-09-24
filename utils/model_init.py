# get architecture
def get_arch(opt):
    arch = opt.arch

    print('You choose ' + arch + '...')
    if arch == 'CNUNet_TB':
        from models.CNUNet_TB import CNUNet_TB
        model = CNUNet_TB(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                          WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)

    elif arch == 'Stage1':
        from models.Stage1 import Stage1
        model = Stage1(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                       WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)

    elif arch == 'Stage2':
        from models.Stage2 import Stage2
        model = Stage2(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                        WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)

    elif arch == 'NUNet-TLS':
        from models.NUNet_TLS import NUNet_TLS
        model = NUNet_TLS(in_ch=opt.in_ch, mid_ch=opt.mid_ch, out_ch=opt.out_ch,
                     WIN_LEN=opt.win_len, HOP_LEN=opt.hop_len, FFT_LEN=opt.fft_len)
    else:
        raise Exception("Arch error!")

    return model


# get trainer and validator (train method)
def get_train_mode(opt):
    loss_type = opt.loss_type
    stage1_training = opt.stage1_training
    stage2_training = opt.stage2_training

    arch = opt.arch

    print('You choose ' + loss_type + '...')

    if arch == 'NUNet-TLS':
        from .trainer import mag_real_imag_loss_real_input_train
        from .trainer import mag_real_imag_loss_real_input_valid
        trainer = mag_real_imag_loss_real_input_train
        validator = mag_real_imag_loss_real_input_valid
    else:
        if loss_type == 'mag':  # multiple(joint) loss function
            from .trainer import mag_loss_train
            from .trainer import mag_loss_valid
            trainer = mag_loss_train
            validator = mag_loss_valid
        elif loss_type == 'mag+real+imag':  # multiple(joint) loss function
            if stage1_training:
                if stage2_training:
                    raise Exception("Training setup error!")
                else:
                    from .trainer import mag_real_imag_loss_stage1_train
                    from .trainer import mag_real_imag_loss_stage1_valid
                    trainer = mag_real_imag_loss_stage1_train
                    validator = mag_real_imag_loss_stage1_valid
            elif stage2_training:
                from .trainer import mag_real_imag_loss_stage2_train
                from .trainer import mag_real_imag_loss_stage2_valid
                trainer = mag_real_imag_loss_stage2_train
                validator = mag_real_imag_loss_stage2_valid
            else:
                from .trainer import mag_real_imag_loss_train
                from .trainer import mag_real_imag_loss_valid
                trainer = mag_real_imag_loss_train
                validator = mag_real_imag_loss_valid
        else:
            raise Exception("Loss type error!")

    return trainer, validator


def get_loss(opt):
    from torch.nn import L1Loss
    from torch.nn.functional import mse_loss
    loss_oper = opt.loss_oper

    print('You choose loss operation with ' + loss_oper + '...')
    if loss_oper == 'l1':
        loss_calculator = L1Loss()
    elif loss_oper == 'l2':
        loss_calculator = mse_loss
    else:
        raise Exception("Arch error!")

    return loss_calculator
