from Inform_project_new.data.echosounder_data.load_data import data_paths  
from Inform_project_new.data.echosounder_data.load_data.echogram import Echogram

import os
import numpy as np

def get_echograms(years, tuple_frequencies, minimum_shape):
    """
    Returns up to 10 echograms from the specified years that match the given frequencies
    and meet minimum shape constraints.
    """

    path_to_echograms = data_paths.path_to_echograms()
    raw_eg_names = os.listdir(path_to_echograms)

    echograms = []
    
    # Normalize year filter
    if years != 'all' and not isinstance(years, (list, tuple, np.ndarray)):
        years = [years]

    for name in raw_eg_names:
        if '.' in name:
            continue

        label_path = os.path.join(path_to_echograms, name, 'labels_heave.dat')
        if not os.path.isfile(label_path):
            continue

        try:
            e = Echogram(os.path.join(path_to_echograms, name))
        except Exception as err:
            print(f"Failed to load echogram {name}: {err}")
            continue

        # Apply all filters
        if (e.shape[0] > minimum_shape and
            e.shape[1] > minimum_shape and
            e.shape[1] == e.time_vector.shape[0] and
            e.shape[1] == e.heave.shape[0] and
            tuple(e.frequencies) == tuple_frequencies and
            (years == 'all' or e.year in years)):
            
            echograms.append(e)

        if len(echograms) == 10:
            break

    return echograms

# Count the number of classes in the echograms
def count_classes_in_echograms(echograms):
    total_class_counts = defaultdict(int)

    for e in echograms:
        labels = e.label_memmap()
        unique_classes, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            total_class_counts[cls] += count

    return total_class_counts
