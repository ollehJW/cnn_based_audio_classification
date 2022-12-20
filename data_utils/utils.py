import numpy as np
import librosa
import os
from collections import defaultdict
from .spectrogram import mel_spectrogram

def construct_filesets(file_path):
    """Construct filesets

    Parameters
    ----------
    train_path : str
        path of train dataset

    Returns
    -------
    files : list
        list of files
    uni_label : list
        unique label list, list of classes
    y : list
        list of one hot encoded arraies
    """
    
    files = []
    for ty in os.listdir(file_path):
        filelist = os.listdir(os.path.join(file_path, ty))
        for i, file in enumerate(filelist):
            if file.endswith('.wav'):
                files.append(os.path.join(file_path, ty, file))

    labels = [file.split('/')[-2] for file in files]
    uni_label = np.unique(labels)

    print("Unique Target Labels: {}".format(uni_label))
    y = np.array([np.eye(len(uni_label))[np.where(uni_label==label)].reshape(-1) for label in labels])

    label_counts = np.sum(y, axis = 0)
    for index in range(len(uni_label)):
        print("{} Counts: {}".format(uni_label[index], int(label_counts[index])))

    return files, uni_label, y


def load_wav(audio_path,sr):
    """Load Wav file format

    Parameters
    -----------
    audio_path : str
        path to wav file
    sr : int
        sampling ratio

    Returns
    -------
    np.array
        raw signal
    """
    signal, _ = librosa.load(audio_path, sr)
    return signal

def make_npy_files(audio_path, sr, img_height, img_width):
    """Generate spectrogram npy files per audio file

    Parameters
    -----------
    audio_path : str
        path to wav file
    sr : int
        sampling rate of signal
    img_height : int 
        target image height
    img_width : int 
        target image width

    Returns
    -------
    .npy file
        melspectrogram. with shape of 2 dimensional numpy array
        (img_height, img_width)
    """

    signal = load_wav(audio_path, sr)
    spec = mel_spectrogram(signal, sr, img_height, img_width)
    save_path = audio_path.replace('audio_files', 'npy_files')
    save_path = save_path.replace('wav', 'npy')
    np.save(save_path, spec)


def make_fileset_indexing(file_name_list, category_list, file_label_list, seed, mixup = False):
    """Make a list of indices for mixup combination

    Parameters
    ------------
    file_name_list : list
        list of filenames
    category_list : list
        cotegory of targets 
    file_label_list : list
        list of one-hot encoded labels corresponds to file_name_list
    seed : int
        Random Seed number
    mixip : bool
        Whether mixup augmentation

    Returns
    --------
    list
        list of mixup indices
    """
    np.random.seed(seed)

    category_index = defaultdict(list)
    for cat in category_list:
        category_index[cat] = list(np.where(file_label_list[:, list(category_list).index(cat)] == 1)[0])
    
    
    dom_cls_idx = np.argmax(np.sum(file_label_list, axis=0))
    dom_cls_name = category_list[dom_cls_idx]
    # print("Dominent Class: {}".format(dom_cls_name))

    list_wo_dom = np.delete(category_list, dom_cls_idx)

    fileset_indexing = [[x1] for x1 in list(range(len(file_name_list)))]

    if mixup:
        for cat in list_wo_dom:
            if len(category_index[cat]) > 0:
                aug_num = len(category_index[dom_cls_name]) - len(category_index[cat])
                dom_sample = np.random.choice(category_index[dom_cls_name], aug_num, replace=True)
                submissive_sample = np.random.choice(category_index[cat], aug_num, replace=True)
                for i in range(aug_num):
                    fileset_indexing.append([submissive_sample[i], dom_sample[i]])

    return fileset_indexing
    








