#!/usr/bin/env python3

# TODO hdf5 overwrite mode, group hierarchy, maybe cache silent identifier

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

from itertools import product as product

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
    pose_from_sfd_tag,
    parse_loop_iter,
    np_rt_from_residues,
    silent_tag_to_poselets,
)


def peptide_dist(res_1, res_2):
    """
    Returns dist C of res_1 to N of res_2

    Intentionally brittle, meant to throw RuntimeError if there's no C/N atoms
    """

    return (res_1.atom("C").xyz() - res_2.atom("N").xyz()).norm()


def parse_xforms_from_poselets(poselets, tag, max_n_mer=20):
    """
    Using 2 res poses to parse all xforms

    We're doing this weird hack because loading poses is big O N^2 on num res,
    so it is programmatically faster to load n/2 poses of size 2 than to load
    one pose of size n on average. If you're reading this docstring and this is
    no longer true, please for goodness sake delete this horror
    """
    n_poselets = len(poselets)
    logging.debug(f"pose size (n_poselets): {n_poselets}")
    xforms_list = []
    loop_data_string_list = []
    for root_i in range(n_poselets):
        pose_1 = poselets[root_i]
        # check for chainbreak, also a proxy for broken pose
        # We can't do all the broken pose checks here, but it filters some egregious things early
        r1 = pose_1.residue(1)
        try:
            dist = peptide_dist(r1, pose_1.residue(2))
            if dist > 1.5:
                # Skip if chainbreak b/w res1/2
                continue
        except RuntimeError as e:
            print(e)
            print(f"error checking poselet: {root_i}")
            print(f"sequence: {poselets[root_i].annotated_sequence()}")
            # if this poselet breaks, skip all rooted on it
            continue
        break_next = False
        for n_mer_end in range(2, max_n_mer):

            if break_next:
                break_next = False
                break
            r2_i = root_i + n_mer_end
            if r2_i >= n_poselets:
                break
            pose_2 = poselets[r2_i]
            r2 = pose_2.residue(1)
            try:
                dist = peptide_dist(r2, pose_2.residue(2))
                if dist > 1.5:
                    # If we detect a chainbreak, we can't take any more stretches from the current root
                    break_next = True

            except RuntimeError as e:
                print(e)
                print(f"error checking poselet: {r2_i}")
                print(f"sequence: {poselets[r2_i].annotated_sequence()}")
                # if this poselet breaks, skip it and all after sharing the same root
                break
            logging.debug("attempting to process xform ...")
            logging.debug(f"start: {root_i +1} end: {r2_i + 1}")

            if not (
                r1.type().is_canonical_aa() and r2.type().is_canonical_aa()
            ):

                logging.debug("invalid anchor residues, not canonical AA")
                continue
            xform = np_rt_from_residues(r1, r2)
            xforms_list.append(xform)
            # small redundancy here on the tag, maybe remove
            # problem is that it's so convenient to just staple these bad boys
            # into a big list as line items later
            loop_data_string_list.append(f"{tag}:{root_i}:{r2_i}")
    return loop_data_string_list, xforms_list


def setup_xbin_vars(xbin_cart_list, xbin_ori_list, file):
    """
    Just here to make main look cleaner
    """

    if file:
        with open(file, "r") as f:
            file_list = f.read().splitlines()

        xbin_cart_list_additional, xbin_ori_list_additional = zip(
            *[tuple(map(float, l.split())) for l in file_list]
        )
        xbin_cart_list.extend(xbin_cart_list_additional)
        xbin_ori_list.extend(xbin_ori_list_additional)
        xbin_cart_list = list(set(xbin_cart_list))
        xbin_ori_list = list(set(xbin_ori_list))
    if not xbin_cart_list:
        xbin_cart_list = [1.0]
    if not xbin_ori_list:
        xbin_ori_list = [15.0]

    return xbin_cart_list, xbin_ori_list


def parse_xforms_from_tag(sfd, tag, max_n_mer=20):
    """
    Shaving some sort of large animal...

    This is wrapper written to compare poselets to full pose loading
    Unfortunately loading poses two res at a time n times (where n is pose len)
    is faster than loading the pose once WTF just rosetta things
    """
    pose = pose_from_sfd_tag(sfd, tag)
    loop_data_string_list = []
    xforms_list = []
    for begin, end in parse_loop_iter(pose):
        if begin < 1 or end > pose.size():
            continue
        r1 = pose.residue(begin)
        r2 = pose.residue(end)
        logging.debug("attempting to process xform ...")
        logging.debug(f"start: {begin} end: {end}")

        if not (r1.type().is_canonical_aa() and r2.type().is_canonical_aa()):

            logging.debug("invalid anchor residues, not canonical AA")
            continue
        xform = np_rt_from_residues(r1, r2)
        xforms_list.append(xform)
        # small redundancy here on the tag, maybe remove
        # problem is that it's so convenient to just staple these bad boys
        # into a big list as line items later
        loop_data_string_list.append(f"{tag}:{begin}:{end}")
    return loop_data_string_list, xforms_list


# using ifmain for click
@click.command()
@click.argument("silent_file", nargs=1)
@click.option("-r", "--rosetta-flags", "rosetta_flags_file", default="")
@click.option(
    "-c",
    "--xbin-cart-res",
    "xbin_cart_list",
    default=[],
    type=float,
    multiple=True,
    help="cart resl to use for xbin. Default is 1. This option can be used multiple times to make a combination of tables",
)
@click.option(
    "-o",
    "--xbin-ori-res",
    "xbin_ori_list",
    default=[],
    type=float,
    multiple=True,
    help="ori resl to use for xbin. Default is 1. This option can be used multiple times to make a combination of tables",
)
@click.option(
    "-m",
    "--max-frag-len",
    "max_len",
    default=20,
    type=int,
    show_default=True,
    help="the max fragment size to use for building the table",
)
@click.option(
    "-f",
    "--resolution-scan-config-file",
    "scan_file",
    type=click.Path(exists=True),
    help="It is possible to provide a file with cart resl in the first column and ori in the second. This will build all the appropriate hashmaps as well as indexing the keys,values, and strings in a single hdf5 archive. If you provide a file and also specify resolutions on command line this script will use both.",
)
def main(
    silent_file,
    rosetta_flags_file="",
    xbin_cart_list=[],
    xbin_ori_list=[],
    max_len=20,
    scan_file="",
):

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

    key_type = np.dtype("i8")
    value_type = np.dtype("i8")
    gp_dict = gp.Dict(key_type, value_type)

    xforms = []
    loop_data_string_list = []

    # silent_name = "loop_structs_out.silent"
    sfd = SilentFileData(
        silent_file, False, False, "binary", SilentFileOptions()
    )

    # for pose in poses_from_silent(silent_file):
    sfd.read_file(silent_file)
    for tag in sfd.tags():
        logging.debug(f"working on tag: {tag}")
        poses = silent_tag_to_poselets(silent_file, tag, 1, 2)
        tag_loop_data_list, tag_xforms_list = parse_xforms_from_poselets(
            poses, tag, max_n_mer=max_len
        )
        # tag_loop_data_list, tag_xforms_list = parse_xforms_from_tag(
        #     sfd, tag, max_n_mer=max_len
        # )
        loop_data_string_list.extend(tag_loop_data_list)
        xforms.extend(tag_xforms_list)

        logging.debug("loop data loaded")
    xbin_cart_list, xbin_ori_list = setup_xbin_vars(
        xbin_cart_list, xbin_ori_list, scan_file
    )
    logging.debug(f"fragments extracted, building tables")
    logging.debug(
        f"""xbin params:
    c:{xbin_cart_list}
    o:{xbin_ori_list}"""
    )
    for xbin_cart, xbin_ori in product(xbin_cart_list, xbin_ori_list):

        binner = xb(cart_resl=xbin_cart, ori_resl=xbin_ori)
        all_keys_non_unique = binner.get_bin_index(np.array(xforms))

        keys_unique = {}
        for key, loop_data_string in zip(
            all_keys_non_unique, loop_data_string_list
        ):
            if key in keys_unique.keys():
                keys_unique[key].append(loop_data_string)
            else:
                keys_unique[key] = [loop_data_string]

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

        gp_dump = f"gp_c{xbin_cart}_o{xbin_ori}.bin"
        gp_dict.dump(gp_dump)

        key_val_data = np.empty((gp_keys.shape[0], 3))
        key_val_data[:, 0] = gp_keys
        key_val_data[:, 1:] = general_vals

        # string_master_file = f"loop_tag_index_c{xbin_cart}_o{xbin_ori}.txt"
        np_strings_master = np.array(strings_master, dtype=np.string_)
        #
        # with open(string_master_file, mode="wt", encoding="utf-8") as f:
        #     f.write("\n".join(strings_master))
        #     f.write("\n")

        # npz_out = "key_val_data.npz"
        # np.savez(npz_out, key_val_data)

        hdf5 = h5py.File("fragment_data.hf5", "a")
        kv_group = hdf5.require_group("key_value_data")
        key_val_ds = kv_group.require_dataset(
            f"key_val_index_cart_{xbin_cart}_ori_{xbin_ori}_nmer_{max_len}",
            key_val_data.shape,
            dtype=key_val_data.dtype,
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
        hdf5.close()


if __name__ == "__main__":
    main()
