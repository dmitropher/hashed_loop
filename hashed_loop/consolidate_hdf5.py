#!/usr/bin/env python3
import click
import numpy as np
import getpy as gp
import pyrosetta

import logging

# import faulthandler

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
@click.option(
    "-o",
    "--output-path",
    "consolidated_path",
    default="consolidated.hf5",
    show_default=True,
    help="pick a different filename to output the consolidated hdf5 to",
)
def main(hdf5_list, silent_list, consolidated_path="consolidated.hf5"):
    ""
    keys_unique = {}
    with open(hdf5_list, "r") as f:
        hdf5_paths = f.read().splitlines()

    max_len = 0
    logging.debug(f"{hdf5_paths}")
    for hdf5_path in hdf5_paths:
        try:
            hdf5_input = h5py.File(hdf5_path, "r")
        except OSError as e:
            logging.error(e)
            continue
        kv_group = hdf5_input["key_value_data"]
        ds_list = []
        logging.debug(f"processing: {hdf5_path}")
        for name in kv_group.keys():
            ds = kv_group[name]
            if not isinstance(ds, h5py.Dataset):

                continue
            else:
                ds_list.append(ds)
        logging.debug("loading datasets into memory")
        for ds in ds_list:
            xbin_cart = ds.attrs["cart_resl"]
            xbin_ori = ds.attrs["ori_resl"]
            frag_ds = retrieve_string_archive(hdf5_input, xbin_cart, xbin_ori)
            if frag_ds is None:
                continue
            ds_max_len = frag_ds.attrs["max_len"]
            if ds_max_len > max_len:
                max_len = ds_max_len
            frags = [s.decode("UTF-8") for s in frag_ds[:]]

            keys = np.array(ds[:, 0], dtype=np.int64)
            vals = np.array(ds[:, 1:], dtype=np.int64)

            keys_loops_iter = (
                (key, frags[val[0] : val[0] + val[1]])
                for (key, val) in zip(keys, vals)
            )
            # This is a key to keys unique! the name is annoying, sorry
            unique_kd_key = (xbin_cart, xbin_ori)

            if unique_kd_key in keys_unique.keys():
                update_unique_key_dict(
                    keys_unique[unique_kd_key], keys_loops_iter
                )
            else:
                keys_unique[unique_kd_key] = {}
                update_unique_key_dict(
                    keys_unique[unique_kd_key], keys_loops_iter
                )
        hdf5_input.close()

    for (cart, ori), keys_unique_sub in keys_unique.items():
        logging.debug(
            f"starting hashmap population for cart: {cart} ori:{ori}"
        )
        offset = 0
        strings_master = []
        gp_vals_list = []
        gp_keys_list = []
        for key, strings in keys_unique_sub.items():
            num_strings = len(strings)
            strings_master.extend(strings)
            gp_keys_list.append(key)
            gp_vals_list.append([offset, num_strings])
            offset += num_strings
            # logging.debug(f"new offset: {offset}")
        gp_keys = np.array(gp_keys_list)
        gp_vals = np.array(gp_vals_list)

        logging.debug("gp_dict data built")
        logging.debug(f"gp_vals: {gp_vals}")

        # Need to flatten the 2 part data (index/offset) into a single number
        # gp_vals_i32_flat = gp_vals.astype(np.int32).reshape(-1)
        # gp_vals_i64 = gp_vals_i32_flat.view(np.int64)

        # key_type = np.int64
        # value_type = np.int64
        # gp_dict = gp.Dict(key_type, value_type)
        # gp_dict[gp_keys] = gp_vals_i64

        # logging.debug("gp_dict object built")
        # gp_dump = "getpy_dict.bin"
        # gp_dict.dump(gp_dump)

        key_val_data = np.empty((gp_keys.shape[0], 3))
        key_val_data[:, 0] = gp_keys
        key_val_data[:, 1:] = gp_vals

        logging.debug("writing all stores to disk")
        hdf5 = h5py.File(consolidated_path, "a")
        kv_group = hdf5.require_group("key_value_data")
        kv_ds_name = (
            f"key_val_index_cart_{xbin_cart}_ori_{xbin_ori}_nmer_{max_len}"
        )
        logging.debug(f"{kv_ds_name}")
        key_val_ds = kv_group.create_dataset(
            kv_ds_name, key_val_data.shape, dtype=key_val_data.dtype
        )
        key_val_ds[:] = key_val_data
        key_val_ds.attrs.create("cart_resl", data=xbin_cart)
        key_val_ds.attrs.create("ori_resl", data=xbin_ori)
        key_val_ds.attrs.create("max_len", data=max_len)
        key_val_ds.attrs.create(
            "description",
            data="this is a 2XN np array where col1 are the int64 xbin keys and col2/3 are two upcast int32s (row,n_strings) addressing a sequence of strings in a corresponding archive",
        )
        str_group = hdf5.require_group("string_archive")

        np_strings_master = np.array(strings_master, dtype=np.string_)
        string_archive = str_group.require_dataset(
            f"string_archive_cart_{xbin_cart}_ori_{xbin_ori}_nmer_{max_len}",
            np_strings_master.shape,
            dtype=np_strings_master.dtype,
        )
        string_archive[:] = np_strings_master
        string_archive.attrs.create("cart_resl", data=xbin_cart)
        string_archive.attrs.create("ori_resl", data=xbin_ori)
        string_archive.attrs.create("max_len", data=max_len)
        string_archive.attrs.create(
            "description",
            data="Ordered archive of strings describing pose fragments by tag:start:end. The order is addressed by the corresponding key_val dataset",
        )
        # str_id = string_archive.id
        # logging.debug(f"str_id: {str_id}")
        # logging.debug(type(str_id))
        # key_val_ds.attrs.create("string_archive_id", data=str_id)
        hdf5.close()

        # string_master_file = "loop_tag_index.txt"
        #
        # with open(string_master_file, mode="wt", encoding="utf-8") as f:
        #     f.write("\n".join(strings_master))
        #     f.write("\n")

        # npz_out = "key_val_data.npz"
        # np.savez(npz_out, key_val_data)

    combine_silents(silent_list, "loop_archive.silent")


if __name__ == "__main__":
    main()
