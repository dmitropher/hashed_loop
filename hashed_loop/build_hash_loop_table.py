#!/usr/bin/env python3
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


import click


import h5py
import numpy as np

# AP Moyer
import getpy as gp

# maybe no NeRF necessary
# from nerf import iNeRF, NeRF

# Will Sheffler
# from homog import hstub
from xbin import XformBinner as xb

# No direct import of pyrosetta needed for this executable
# import pyrosetta
from pyrosetta.rosetta.core.io.silent import SilentFileData as SilentFileData
from pyrosetta.rosetta.core.io.silent import (
    SilentFileOptions as SilentFileOptions,
)

from hashed_loop import (
    check_no,
    dumb_clean_pose,
    run_pyrosetta_with_flags,
    poses_from_silent,
    parse_loop_iter,
    np_rt_from_residues,
)

# using ifmain for click
@click.command()
@click.argument("silent_file", nargs=1)
@click.option("-r", "--rosetta-flags", "rosetta_flags_file", default="")
@click.option(
    "-c",
    "--xbin-cart-res",
    "xbin_cart",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "-o",
    "--xbin-ori-res",
    "xbin_ori",
    default=15,
    type=float,
    show_default=True,
)
def main(silent_file, rosetta_flags_file="", xbin_cart=1.0, xbin_ori=15.0):

    """
    Generates a loop e2e xbin table referencing a silentfile with metadata

    e2e hashtable holds just xbin keys and table entry keys. The table entry
    keys are just arbitrary indices that go to a table referncing a silentfile

    The table is just a list of tags and positions in a text list. The values
    deposited in the getpy dict are a two number coordinate: start index and num entries.
    To find all the appropriate values, just load the txt list into a python list
    and slice it: list[start:start + num_entries].


    """
    run_pyrosetta_with_flags(rosetta_flags_file)

    binner = xb(cart_resl=xbin_cart, ori_resl=xbin_ori)

    key_type = np.dtype("i8")
    value_type = np.dtype("i8")
    gp_dict = gp.Dict(key_type, value_type)

    xforms = []
    loop_data_list = []

    silent_name = "loop_structs_out.silent"
    sfd_out = SilentFileData(
        silent_name, False, False, "binary", SilentFileOptions()
    )

    for pose in poses_from_silent(silent_file):

        logging.debug("pose loaded")
        tag = pose.pdb_info().name()

        logging.debug(f"working on tag: {tag}")

        # print(pose.size())
        # pose.dump_pdb("pre_ala_debug.pdb")
        logging.debug("checking for N O")

        if not (check_no(pose)):
            continue

        logging.debug("no broken CA found, attempting to clean ligands")
        try:
            pose = dumb_clean_pose(pose)
        except RuntimeError as e:
            print(e)
            print("skipping pose")
            continue

        logging.debug("dumb clean complete")

        struct = sfd_out.create_SilentStructOP()
        struct.fill_struct(pose, tag)
        sfd_out.add_structure(struct)
        sfd_out.write_silent_struct(struct, silent_name, False)

        logging.debug("struct written to silent")

        for start, end in parse_loop_iter(pose):
            start_res = pose.residue(start)
            end_res = pose.residue(end)
            logging.debug("attempting to process xform ...")
            logging.debug(f"start: {start} end: {end}")

            if not (
                start_res.type().is_canonical_aa()
                and end_res.type().is_canonical_aa()
            ):

                logging.debug("invalid anchor residues, not canonical AA")
                continue
            e2e_xform = np_rt_from_residues(
                pose.residue(start), pose.residue(end)
            )
            loop_data_list.append((tag, start, end))
            xforms.append(e2e_xform)
            logging.debug("xforms processed")

    logging.debug("loop data loaded")

    all_keys_non_unique = binner.get_bin_index(np.array(xforms))

    keys_unique = {}
    for key, loop_data in zip(all_keys_non_unique, loop_data_list):
        if key in keys_unique.keys():
            keys_unique[key].append(
                f"{loop_data[0]}:{loop_data[1]}:{loop_data[2]}"
            )
        else:
            keys_unique[key] = [
                f"{loop_data[0]}:{loop_data[1]}:{loop_data[2]}"
            ]

    offset = 0
    strings_master = []
    gp_vals_list = []
    gp_keys_list = []
    for key, strings in keys_unique.items():
        num_strings = len(strings)
        strings_master.extend(strings)
        gp_keys_list.append(key)
        gp_vals_list.append([offset, num_strings])
        offset += num_strings
    gp_keys = np.array(gp_keys_list)
    gp_vals = np.array(gp_vals_list)
    general_vals = gp_vals
    # squash data to fit into getpy_dict
    gp_vals = gp_vals.astype(np.int32).reshape(-1)
    gp_vals = gp_vals.view(np.int64)

    gp_dict[gp_keys] = gp_vals

    gp_dump = "getpy_dict.bin"
    gp_dict.dump(gp_dump)

    key_val_data = np.empty((gp_keys.shape[0], 3))
    key_val_data[:, 0] = gp_keys
    key_val_data[:, 1:] = general_vals

    string_master_file = "loop_tag_index.txt"

    with open(string_master_file, mode="wt", encoding="utf-8") as f:
        f.write("\n".join(strings_master))
        f.write("\n")

    npz_out = "key_val_data.npz"
    np.savez(npz_out, key_val_data)

    with h5py.File("key_val_data.hf5", "w") as hdf5:
        hdf5.create_dataset(
            "n_mer_key_val_index",
            key_val_data.shape,
            dtype=key_val_data.dtype,
            data=key_val_data,
        )


if __name__ == "__main__":
    main()
