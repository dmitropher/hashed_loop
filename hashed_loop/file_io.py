import os

import pyrosetta


def safe_load_pdbs(pdbs):
    for pdb in pdbs:
        try:
            yield pyrosetta.pose_from_pdb(pdb)
        except RuntimeError as e:
            print(e)
            print(f"unable to load: {pdb}")
            continue


def default_hdf5():
    """
    Return the hashed_loop default HDF5 path

    Do not mess with the HDF5 at this path unless you know what you are doing
    """
    hdf5_path = os.path.join(
        os.path.dirname(__file__), "resources/hdf5_archives/default.hf5"
    )
    return hdf5_path


def default_silent():
    """
    Return the hashed_loop default silent path

    Do not mess with the silent file at this path unless you know what you are doing
    """
    silent_path = os.path.join(
        os.path.dirname(__file__), "resources/silent_files/default.silent"
    )
    return silent_path


def retrieve_string_archive(hdf5, xbin_cart, xbin_ori):
    """
    Returns the matching string archive ds by xbin params
    """
    # TODO chance archives to store the matching DatasetID instead of this mess
    string_archives = hdf5.require_group("string_archive")
    for key in string_archives.keys():
        strings_ds = string_archives[key]
        cart, ori = (
            strings_ds.attrs[name] for name in ("cart_resl", "ori_resl")
        )
        if cart == xbin_cart and ori == xbin_ori:
            return strings_ds
