import h5py


def get_omics_sizes_from_dataset(hdf5_file):
    category_counts = {}
    with h5py.File(hdf5_file, 'r') as f:
        first_case_id = list(f.keys())[0]  # Get the first case ID
        omics_group = f[first_case_id]['omics']  # Access the 'omics' group under the first case ID
        for category in omics_group.keys():
            category_counts.setdefault(category, 0)
            category_counts[category] = len(omics_group[category])  # Count number of values in each category
    sorted_counts = [category_counts[category] for category in sorted(category_counts.keys())]
    return sorted_counts
