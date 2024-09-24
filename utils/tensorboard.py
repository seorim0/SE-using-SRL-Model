from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pylab as plt
import librosa
import io
import PIL


class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)

    def log_train_loss(self, loss_type, train_loss, step):
        self.add_scalar('train_{}_loss'.format(loss_type), train_loss, step)

    def log_valid_loss(self, loss_type, valid_loss, step):
        self.add_scalar('valid_{}_loss'.format(loss_type), valid_loss, step)

    def log_score(self, metrics_name, metrics, step):
        self.add_scalar(metrics_name, metrics, step)

    def log_wav(self, noisy_wav, clean_wav, enhanced_wav, step):
        # <Audio>
        self.add_audio('noisy_wav', noisy_wav, step, sample_rate=16000)
        self.add_audio('clean_target_wav', clean_wav, step, sample_rate=16000)
        self.add_audio('enhanced_wav', enhanced_wav, step, sample_rate=16000)

    def log_spectrogram(self, noisy_wav, clean_wav, enhanced_wav, step):
        # <Audio>
        self.plot_spectrogram('noisy_wav_spectrogram', noisy_wav, step, sample_rate=16000)
        self.plot_spectrogram('clean_target_wav_spectrogram', clean_wav, step, sample_rate=16000)
        self.plot_spectrogram('enhanced_wav_spectrogram', enhanced_wav, step, sample_rate=16000)

    def plot_spectrogram(self, tag, wav, step, sample_rate=16000):
        wav = wav.cpu().numpy()

        # Convert wav to spectrogram
        D = librosa.stft(wav)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Plot
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Log to TensorBoard
        self.add_image(tag, np.array(PIL.Image.open(buf)), step, dataformats='HWC')

        # Close plot
        plt.close()

