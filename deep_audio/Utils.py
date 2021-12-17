def print_len(arrs):
    lengths = lambda x: len(x)
    print(lengths(arrs))


def combine_params(dicts):
    from itertools import product

    keys, values = zip(*dicts.items())
    return [dict(zip(keys, v)) for v in product(*values)]
