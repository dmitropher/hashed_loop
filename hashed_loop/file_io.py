import os
import math
import h5py
import pyrosetta
import json
import getpy as gp
import numpy as np

# TODO gp_dict caching in resources
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


def get_sorted_ds_list(hdf5):
    kv_group = hdf5["key_value_data"]
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


def cache_gp_dict(gp_dict, ori, cart):
    """
    Caches the Parallel Hashmap for future runs

    Makes no distinction between key/value type in the hashmap
    """
    dest_path_gp_cache_dir = os.path.join(
        os.path.dirname(__file__), "cache/gp_dicts/"
    )
    os.makedirs(dest_path_gp_cache_dir, exist_ok=True)
    name = f"o{ori}_c{cart}.bin"
    gp_dict.dump(dest_path_gp_cache_dir + name)
    key = f"{(ori, cart)}"
    try:
        with open(dest_path_gp_cache_dir + "/hashmaps.json", "r") as f:
            index_dict = json.load(f)
        index_dict[key] = name
    except FileNotFoundError:
        index_dict = {key: name}
    except json.decoder.JSONDecodeError:
        index_dict = {key: name}

    with open(dest_path_gp_cache_dir + "/hashmaps.json", "w") as f:
        json.dump(index_dict, f)


def retrieve_gp_dict_from_cache(ori, cart, key_type, value_type):
    """
    Checks cache for gp_dict, returns None if not found
    """
    dest_path_gp_cache_dir = os.path.join(
        os.path.dirname(__file__), "cache/gp_dicts/"
    )
    os.makedirs(dest_path_gp_cache_dir, exist_ok=True)

    key = f"{(ori, cart)}"
    try:
        with open(dest_path_gp_cache_dir + "/hashmaps.json", "r") as f:
            index_dict = json.load(f)
        name = index_dict.get(key)
        if name is None:
            return
    except FileNotFoundError:
        return
    except json.decoder.JSONDecodeError:
        return

    gp_dict = gp.Dict(key_type, value_type)
    gp_dict.load(dest_path_gp_cache_dir + "/" + name)
    return gp_dict


def build_gp_dict(data_set, key_type, value_type):
    """
    Convert dataset with int64 keys and int32/int32 vals into int64/int64 gpdict
    """
    gp_dict = gp.Dict(key_type, value_type)

    gp_keys = np.array(data_set[:, 0]).astype(np.int64)
    gp_vals = np.array(data_set[:, 1:]).astype(np.int64)

    gp_vals = gp_vals.astype(np.int32).reshape(-1)
    gp_vals = gp_vals.view(np.int64)

    gp_dict[gp_keys] = gp_vals
    return gp_dict


def build_and_cache_gp_dicts(store_path):
    """
    Reads in all data under 'key_value_data' in the store and caches gp_dicts
    """
    with h5py.File(store_path, "r") as hdf5:
        kv_group = hdf5["key_value_data"]
        for name in kv_group.keys():
            ds = kv_group[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            else:
                ori = ds.attrs["ori_resl"]
                cart = ds.attrs["cart_resl"]
                gp_dict = build_gp_dict(ds, np.int64, np.int64)
                cache_gp_dict(gp_dict, ori, cart)
