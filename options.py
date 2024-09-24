"""
Docstring for Options
"""


class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--nepoch', type=int, default=80, help='training epochs')
        parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.00075, help='initial learning rate')
        parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')

        # train settings
        parser.add_argument('--arch', type=str, default='MSC', help='archtechture')
        parser.add_argument('--loss_type', type=str, default='mag+real+imag', help='loss function type')
        parser.add_argument('--loss_oper', type=str, default='l2', help='loss function operation type')
        parser.add_argument('--c', type=list, default=[0.1, 0.9, 0.2, 0.05], help='coupling constant')
        parser.add_argument('--device', type=str, default='cuda', help='gpu or cpu')

        parser.add_argument('--stage1_training', type=bool, default=False, help='')
        parser.add_argument('--stage2_training', type=bool, default=False, help='')

        # network settings
        parser.add_argument('--in_ch', type=int, default=2, help='channel size for input dim')
        parser.add_argument('--mid_ch', type=int, default=32, help='channel size for middle dim')
        parser.add_argument('--out_ch', type=int, default=64, help='channel size for output dim')

        # pretrained
        parser.add_argument('--env', type=str, default='base', help='log name')
        parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained_weights')
        parser.add_argument('--pretrained_init', type=bool, default=False, help='load pretrained_weights')
        parser.add_argument('--pretrain_model_path', type=str, default='./log/CNUNet_TB/models/chkpt_1.pt',
                            help='path of pretrained_weights')
        parser.add_argument('--test_name', type=str, default='exp1', help='wave write')
        parser.add_argument('--wav_write_flag', type=bool, default=True, help='wave write')

        # dataset
        parser.add_argument('--database', type=str, default='VBD', help='database')
        parser.add_argument('--fft_len', type=int, default=512, help='fft length')
        parser.add_argument('--win_len', type=int, default=512, help='window length')
        parser.add_argument('--hop_len', type=int, default=256, help='hop length')
        parser.add_argument('--fs', type=int, default=16000, help='sampling frequency')
        parser.add_argument('--chunk_size', type=int, default=32000, help='chunk size')

        parser.add_argument('--noisy_dirs_for_train', type=str,
                            default='../Dataset/VBD/train/noisy/',
                            help='noisy dataset addr for train')
        parser.add_argument('--noisy_dirs_for_valid', type=str,
                            default='../Dataset/VBD/test/noisy/',
                            help='noisy dataset addr for valid')
        parser.add_argument('--noisy_dirs_for_test', type=str,
                            default='../Dataset/VBD/test/noisy/',
                            help='noisy dataset addr for test')

        return parser
