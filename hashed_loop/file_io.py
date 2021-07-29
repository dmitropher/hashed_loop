import os
import math
import h5py
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


def get_sorted_ds_list(hdf5_handle):
    """
    Returns a list of datasets sorted by ori and cart from finest to coarsest
    """
    kv_group = hdf5_handle["key_value_data"]
    ds_list = []
    for name in kv_group.keys():
        ds = kv_group[name]
        if not isinstance(ds, h5py.Dataset):
            continue
        else:
            ds_list.append(ds)

    cart_avg, ori_avg = (
        sum(resls) / len(resls)
        for resls in zip(
            *((ds.attrs["cart_resl"], ds.attrs["ori_resl"]) for ds in ds_list)
        )
    )
    sorted_ds_list = sorted(
        ds_list,
        key=(
            lambda ds: (
                math.sqrt(
                    (ds.attrs["cart_resl"] * ori_avg) ** 2
                    + (ds.attrs["ori_resl"] * cart_avg) ** 2
                )
                / (cart_avg * ori_avg)
            )
        ),
    )
    return sorted_ds_list
