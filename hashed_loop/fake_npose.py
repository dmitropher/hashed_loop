import numpy as np
from npose_util import N as N
from npose_util import C as C
from npose_util import CA as CA
from npose_util import extract_atoms


def get_stubs_from_n_ca_c(n, ca, c):
    e1 = n - ca
    e1 = np.divide(e1, np.linalg.norm(e1, axis=1)[..., None])

    e3 = np.cross(e1, c - ca, axis=1)
    e3 = np.divide(e3, np.linalg.norm(e3, axis=1)[..., None])

    e2 = np.cross(e3, e1, axis=1)

    stub = np.zeros((len(n), 4, 4))
    stub[..., :3, 0] = e1
    stub[..., :3, 1] = e2
    stub[..., :3, 2] = e3
    stub[..., :3, 3] = ca
    stub[..., 3, 3] = 1.0

    return stub


def get_stubs_from_npose(npose):
    ns = extract_atoms(npose, [N])
    cas = extract_atoms(npose, [CA])
    cs = extract_atoms(npose, [C])

    return get_stubs_from_n_ca_c(ns[:, :3], cas[:, :3], cs[:, :3])


def tpose_from_npose(npose):
    return get_stubs_from_npose(npose)


def itpose_from_tpose(tpose):
    return np.linalg.inv(tpose)
