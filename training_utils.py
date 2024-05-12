import os
import numpy as np

from src.utils import load_dict
from src.preprocess import load_np_data, data_process


def get_data(training_data_path, generations, window_size_cM, model_type):
    """ Creating splits from training data
    Imported almost as is from Gnomix repo (gnomix.py)
    The only difference is type casting in concrete mode
    """

    # ------------------ Meta ------------------
    assert(type(generations) == dict), "Generations must be a dict with list of generations to read in for each split"

    laidataset_meta_path = os.path.join(training_data_path, "metadata.pkl")
    laidataset_meta = load_dict(laidataset_meta_path)

    snp_pos = laidataset_meta["pos_snps"]
    snp_ref = laidataset_meta["ref_snps"]
    snp_alt = laidataset_meta["alt_snps"]
    pop_order = laidataset_meta["num_to_pop"]
    pop_list = []
    for i in range(len(pop_order.keys())):
        pop_list.append(pop_order[i])
    pop_order = np.array(pop_list)

    A = len(pop_order)
    C = len(snp_pos)
    M = int(
        round(
            window_size_cM * (C / (100 * laidataset_meta["morgans"]))
        )
    )
    
    
    # Correction to avoid bug
    if C % M == 0:
        M += 1

    meta = {
        "A": A, # number of ancestry
        "C": C, # chm length
        "M": M, # window size in SNPs
        "snp_pos": snp_pos,
        "snp_ref": snp_ref,
        "snp_alt": snp_alt,
        "pop_order": pop_order
    }

    # ------------------ Process data ------------------

    def read(split):
        paths = [os.path.join(training_data_path, split, "gen_" + str(gen)) for gen in generations[split]]
        X_files = [p + "/mat_vcf_2d.npy" for p in paths]
        labels_files = [p + "/mat_map.npy" for p in paths]
        X_raw, labels_raw = [load_np_data(f) for f in [X_files, labels_files]]
        X, y = data_process(X_raw, labels_raw, M)
        return X, y

    X_t1, y_t1 = read("train1")
    X_t2, y_t2 = read("train2")
    X_v, y_v = read("val")

    if model_type == "concrete":
        X_t1 = X_t1.astype(np.float32)
        y_t1 = y_t1.astype(np.int16)
        X_t2 = X_t2.astype(np.float32)
        y_t2 = y_t2.astype(np.int16)
        X_v = X_v.astype(np.float32)
        y_v = y_v.astype(np.int16)

    data = ((X_t1, y_t1), (X_t2, y_t2), (X_v, y_v))

    return data, meta