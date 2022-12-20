
import numpy as np
from copy import copy

def spec_augmentation(spec):
    """ Spec Augmentation to spectrogram

    Parameters
    ----------
    spec : np.array
        spectrogram image
    
    Returns
    -------
    np.array
        augmented spectrogram image    
    """
    time_masking_proportion = np.random.uniform(0, 0.2)
    freq_masking_proportion = np.random.uniform(0, 0.2)
    time_length = np.shape(spec)[1]
    freq_length = np.shape(spec)[0]
    time_masking_length = int(time_length * time_masking_proportion)
    freq_masking_length = int(freq_length * freq_masking_proportion)
    start_time = int(round(np.random.uniform(0, time_length - time_masking_length)))
    start_freq = int(round(np.random.uniform(0, freq_length - freq_masking_length)))
    spec_aug = copy(spec)
    spec_aug[start_freq:(start_freq + freq_masking_length), :] = 0
    spec_aug[:, start_time:(start_time + time_masking_length)] = 0
    return spec_aug
    
def rolling_augmentation(spec, rolling_bound):
    """ Rolling Augmentation to spectrogram

    Parameters
    ----------
    spec : np.array
        spectrogram image
    rolling_bound : float
        rolling augmentation x bounds
        bounds needs to be in [0, 1)
    
    Returns
    -------
    np.array
        augmented spectrogram image    
    """
    assert rolling_bound <= 1, f"rolling bound needs to be less than or equal to 1"
    assert rolling_bound > 0, f"rolling bound needs to be greater than 0"
    
    time = np.shape(spec)[1]
    rolling_point = np.random.uniform(0,rolling_bound)
    left = spec[:,:int(rolling_point*time)]
    right = spec[:,int(rolling_point*time):]
    rolling_aug = np.concatenate((right, left), axis=1)
    return rolling_aug

def mix_up_augmentation(spec1, spec2, spec1_label, spec2_label, mixup_alpha, mixup_beta):
    """ Mixup two spectrogram

    Parameters
    spec1 : np.array 
        spectrogram image 1 
    spec2 : np.array 
        spectrogram image 2 
    spec1_label : np.array
        one-hot encoded label of spectrogram image 1
    spec2_label : np.array
        one-hot encoded label of spectrogram image 2
    mixup_alpha : float
        alpha value of beta distribution for mixup.
    mixup_beta : float
        beta value of beta distribution for mixup.
    Returns
    -------
    tuple
        mixed up image and mixed up label
    """
    lam = np.random.beta(mixup_alpha, mixup_beta)
    lam = np.clip(lam, 0.00001, 0.99999)
    mixup_spec = (1-lam) * spec1 + lam * spec2
    mixup_label = (1-lam) * spec1_label + lam * spec2_label
    return (mixup_spec, mixup_label)