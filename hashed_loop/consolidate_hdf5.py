#!/usr/bin/env python3
import click
import numpy as np
import getpy as gp
import pyrosetta

import logging
import faulthandler

import h5py

from file_io import retrieve_string_archive

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def update_unique_key_dict(key_dict, keys_loops_iter):
    for key, loop_list in keys_loops_iter:
        if key in key_dict.keys():
            key_dict[key].extend(loop_list)
        else:
            key_dict[key] = loop_list
    return key_dict


def poses_from_silent(silent_filename):
    """
    Returns an Iterator object which is composed of Pose objects from a silent file.
    @atom-moyer
    """
    sfd = pyrosetta.rosetta.core.io.silent.SilentFileData(
        pyrosetta.rosetta.core.io.silent.SilentFileOptions()
    )
    sfd.read_file(silent_filename)
    for tag in sfd.tags():
        try:
            ss = sfd.get_structure(tag)
            pose = pyrosetta.rosetta.core.pose.Pose()
            ss.fill_pose(pose)
            pose.pdb_info().name(tag)
            yield pose
        except RuntimeError as e:
            print("issue loading pose, skipping")
            print(e)
            continue


def combine_silents(silent_paths_list, out_path):

    with open(silent_paths_list, "r") as f:
        paths = f.read().splitlines()
    with open(out_path, "wb") as outfile:
        for path in paths:
            with open(path, "rb") as data:
                outfile.write(data.read())


@click.command()
@click.argument("hdf5_list", nargs=1)
@click.argument("silent_list", nargs=1)
def main(dict_list, frag_list, silent_list):
    ""
    keys_unique = {}
    with open(hdf5_list, "r") as f:
        hdf5_paths = f.read().splitlines()
    # with open(frag_list, "r") as f:
    #     frag_paths = f.read().splitlines()
    data_paths_iter = zip(dict_paths, frag_paths)

    for hdf5_path in hdf5_paths:
        hdf5 = h5py.File(hdf5_path, "r")
        kv_group = hdf5["key_value_data"]
        ds_list = []
        for name in kv_group.keys():
            ds = kv_group[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            else:
                ds_list.append(ds)
        frags = f.read().splitlines()

        keys = dict_temp.keys()
        vals = dict_temp[keys].view(np.int32).reshape(-1, 2)

        keys_loops_iter = (
            (key, frags[val[0] : val[0] + val[1]])
            for key, val in zip(keys, vals)
        )

        update_unique_key_dict(keys_unique, keys_loops_iter)

    logging.debug(f"starting hashmap population")
    offset = 0
    strings_master = []
    gp_vals_list = []
    gp_keys_list = []
    for key, strings in keys_unique.items():
        logging.debug(f"key string pair: {key}, {strings}")
        num_strings = len(strings)
        logging.debug(f"num_strings: {num_strings}")
        strings_master.extend(strings)
        gp_keys_list.append(key)
        gp_vals_list.append([offset, num_strings])
        offset += num_strings
        logging.debug(f"new offset: {offset}")
    gp_keys = np.array(gp_keys_list)
    gp_vals = np.array(gp_vals_list)

    logging.debug(f"gp_vals: {gp_vals}")

    gp_vals_i32_flat = gp_vals.astype(np.int32).reshape(-1)

    logging.debug(f"gp_vals_i32_flat: {gp_vals_i32_flat}")

    gp_vals_i64 = gp_vals_i32_flat.view(np.int64)

    key_type = np.int64
    value_type = np.int64
    gp_dict = gp.Dict(key_type, value_type)
    gp_dict[gp_keys] = gp_vals_i64

    gp_dump = "getpy_dict.bin"
    gp_dict.dump(gp_dump)

    key_val_data = np.empty((gp_keys.shape[0], 3))
    key_val_data[:, 0] = gp_keys
    key_val_data[:, 1:] = gp_vals

    string_master_file = "loop_tag_index.txt"

    with open(string_master_file, mode="wt", encoding="utf-8") as f:
        f.write("\n".join(strings_master))
        f.write("\n")

    npz_out = "key_val_data.npz"
    np.savez(npz_out, key_val_data)

    combine_silents(silent_list, "loop_archive.silent")


if __name__ == "__main__":
    main()
