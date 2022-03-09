import pickle



def pickler(obj, fpath, mode='wb'):
    with open(fpath, mode) as f:
        pickle.dump(obj, f)
