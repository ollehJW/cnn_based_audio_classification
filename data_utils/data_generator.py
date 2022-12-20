from torch.utils.data import Dataset
from .utils import make_fileset_indexing
from .augment import mix_up_augmentation, rolling_augmentation, spec_augmentation
from .spectrogram import spec_to_img, standarize
import numpy as np

class ConstructDataset(Dataset):
    """
    Construct pytorch Dataset from file list.
    Parameters
    ----------
    file_list : list
        image file list
    target_list : list
        target list
    phase : str
        train phase. (Default: 'train')
    Returns
    --------
    pytorch Dataset
    """

    def __init__(self, file_list, uni_label, target_list, mixup = False, rolling = False, aug_spec = False):
        self.file_list = file_list
        self.uni_label = uni_label
        self.target_list = target_list
        self.mixup = mixup
        self.rolling = rolling
        self.aug_spec = aug_spec
        

    def file_indexing(self, seed):
        self.file_indices = make_fileset_indexing(self.file_list, self.uni_label, self.target_list, seed = seed, mixup = self.mixup)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Whether to apply each augmentation
        AUG_Prob = np.random.uniform(0,1,3)

        # Batch File Indices
        file_indices = self.file_indices[index]

        if len(file_indices) > 1:
            if self.mixup:
                if AUG_Prob[0] > 0.5:
                    spec1 = np.load(self.file_list[file_indices[0]])
                    spec2 = np.load(self.file_list[file_indices[1]])
                    label1 = self.target_list[file_indices[0]]
                    label2 = self.target_list[file_indices[1]]
                    spec, target = mix_up_augmentation(spec1, spec2, label1, label2, mixup_alpha=1, mixup_beta=7)
                else:
                    spec = np.load(self.file_list[file_indices[0]])
                    target = self.target_list[file_indices[0]]
        else:
            spec = np.load(self.file_list[file_indices[0]])
            target = self.target_list[file_indices[0]]

        if self.rolling:
            if AUG_Prob[1] > 0.5:
                spec = rolling_augmentation(spec, 0.5)
        
        if self.aug_spec:
            if AUG_Prob[2] > 0.5:
                spec = spec_augmentation(spec)

        spec = spec_to_img(spec, n_channel=3)
        spec = standarize(spec)

        return {'spectrogram': spec, 'target': target}



