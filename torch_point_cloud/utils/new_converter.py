import numpy as np

def sparseLabel_to_denseLabelConverter(label):
    unique_label = np.unique(label)
    converted_num = np.arange(0,len(unique_label))
    label_converter = np.full(np.max(unique_label)+1,-1)
    label_converter[unique_label] = converted_num
    return label_converter

def sparseLabel_to_denseLabel(label):
    converter = sparseLabel_to_denseLabelConverter(label)
    converted_label = list(map(lambda x:converter[x],label))
    return converted_label

def batch_sparseLabel_to_denseLabel(labels):
    return np.apply_along_axis(sparseLabel_to_denseLabel, 1, labels)

def label_to_mask(labels, max_label=None):
    # N = labels.shape
    N = len(labels)
    if max_label is None:
        max_label = np.amax(labels)
    mask = np.zeros((N,max_label+1))
    mask[np.arange(N), labels] = 1
    return mask



