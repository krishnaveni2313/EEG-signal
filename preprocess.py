import numpy as np
import scipy.signal
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler as SS

def preprocess_signal(fpath):
    """Preprocess the EEG signal from .mat file."""
    annots = loadmat(fpath)
    try:
        con_list = [[element for element in upperElement] for upperElement in annots['interictal']]
    except:
        con_list = [[element for element in upperElement] for upperElement in annots['ictal']]
    
    Fsignal = np.array(con_list).flatten()

    # Apply Butterworth bandpass filter
    b, a = scipy.signal.butter(3, [0.1, 0.5], btype='band')
    filtered = scipy.signal.filtfilt(b, a, Fsignal)

    # Standardize
    R_filtered = filtered.reshape(-1, 1)
    SC = SS()
    Pre_Data = SC.fit_transform(R_filtered)
    return Pre_Data.flatten()
