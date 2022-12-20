import cv2
from librosa.feature import melspectrogram
import librosa
import numpy as np


def mel_spectrogram(signal, sr, img_height, img_width):
    """Generate melody spectrogram with targeted size

    Parameters
    -----------
    signal : np.array
        1 dimensional (N,) signal
    sr : int
        sampling rate of signal
    img_height : int 
        target image height
    img_width : int 
        target image width

    Returns
    -------
    np.array
        melspectrogram. with shape of 2 dimensional numpy array
        (img_height, img_width)
    """
    
    hop_length = len(signal) // img_width
    n_fft = hop_length * 4 
    spec=melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                        hop_length=hop_length, n_mels=224)
    spec= librosa.power_to_db(S=spec, ref=np.max)
    spec = spec[ :, :img_width]
    spec=cv2.resize(spec,[img_height, img_width])
    return spec

def standarize(img):
    """Normalize a Single Image by Gaussian Distribution (mean, std).

    Parameters
    -----------
    img : np.array
        a single image np array with channel last (C x H x W) 
    Returns
    -------
    np.array
        standarized image
    """
    img = (img - np.mean(img, axis=(1,2))[:, np.newaxis, np.newaxis]) / \
        np.std(img, axis=(1,2))[:, np.newaxis, np.newaxis]
    return img

def spec_to_img(spec, n_channel=3):
    """make 2d np.array to 3d np.array, the last dimension works as a channel
    i.e. original 2d np.array spectrogram (H x W) -->
    image (C x H x W), where C is equal to n_channel

    Parameters
    ----------
    spec : np.array
        2 dimensional numpy array
    n_channel : int
        number of channel, it should be 1 or 3. Defaults to 3.

    Returns
    --------
    np.array 
        dimension increased numpy array, could work as image in 
        matplotlib or tensorflow image
    """
    assert n_channel in [1,3], ValueError("n_channel should be 1 or 3")
    img_1ch = np.expand_dims(spec, axis=0)
    if n_channel ==1 :
        return img_1ch
    else:
        img_nch = np.concatenate([img_1ch]*n_channel, axis=0)
        return img_nch 