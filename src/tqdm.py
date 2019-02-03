_tqdm_disabled = False
_imported_tqdm_func = None


def disable_tqdm():
    global _tqdm_disabled
    _tqdm_disabled = True


def tqdm_wrapper(x, **kwargs):
    if _tqdm_disabled:
        return x
    global _imported_tqdm_func
    if _imported_tqdm_func is None:
        from tqdm import tqdm as tqdm_func
        _imported_tqdm_func = tqdm_func
    return _imported_tqdm_func(x, **kwargs)


tqdm = tqdm_wrapper
