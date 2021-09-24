#! /usr/bin/env python3
import logging
import os

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import click

# import getpy as gp
import numpy as np
from xbin import XformBinner as xb
import pandas as pd
import h5py

from hashed_loop import (
    align_loop,
    run_pyrosetta_with_flags,
    np_rt_from_residues,
    poses_from_silent,
    link_poses,
    sfd_tag_slice,
    get_chains,
    silent_preload,
    subset_bb_rmsd,
    get_closure_hits,
)
from pose_manager import PoseManager as poseman

from hashed_loop.file_io import (
    default_hdf5,
    default_silent,
    safe_load_pdbs,
    get_sorted_ds_list,
)

from pyrosetta.rosetta.utility import vector1_bool as vector1_bool


def preload(rosetta_flags_file):
    """
    Util to preload all the stuff and initialize from user inputs
    """

    run_pyrosetta_with_flags(rosetta_flags_file)

    sfd, silent_index, silent_out = silent_preload(default_silent())
    hdf5_handle = h5py.File(default_hdf5(), "r")

    return hdf5_handle, sfd, silent_index, silent_out


def update_report(
    master_df, closure_quality, attempted_poses, pose_mask, cart_resl, ori_resl
):
    """
    Dump closure data
    """
    tags_closure = np.array(
        [[p.pdb_info().name(), p.chain_end(1)] for p in attempted_poses]
    )
    tags = tags_closure[:, 0]
    closure = tags_closure[:, 1].astype(int)
    targ_plus_1 = closure + 1

    df_dict = {}
    df_dict["pose_name"] = tags
    df_dict["break_start"] = closure
    df_dict["break_end"] = targ_plus_1
    df_dict["closure_found"] = pose_mask
    df_dict["closure_bb_rmsd"] = closure_quality
    df_dict["cart_resl"] = np.full_like(closure_quality, cart_resl)
    df_dict["ori_resl"] = np.full_like(closure_quality, ori_resl)
    df = pd.DataFrame(df_dict)
    df.index.set_names(f"index_c{cart_resl}_o{ori_resl}")
    if master_df.empty:
        return df
    else:
        out = pd.concat([master_df, df])
        return out
    # df.to_csv("closure_data.csv")


def poses_from_paths(*paths, silent_mode=False):
    """
    Returns pose iterator from either pdb or silent paths
    """
    if silent_mode:
        for path in paths:
            yield from poses_from_silent(path)
    else:
        yield from safe_load_pdbs(paths)


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


@click.command()
@click.argument("input_structure_paths", nargs=-1)
@click.option(
    "-r",
    "--rosetta-flags",
    "rosetta_flags_file",
    default="",
    show_default=True,
    help="Rosetta flags file to run pyrosetta with. If unsure, try leaving this blank",
)
# TODO split everything mode into "output low-res also" and "give up after target rmsd reached"
# @click.option(
#     "-e",
#     "--everything-mode",
#     "one_with_everything",
#     is_flag=True,
#     help="Keep outputting until max outputs reached even if a high res solution is found.",
# )
@click.option(
    "-c",
    "--loop-count-per-closure",
    "loop_count_per_closure",
    nargs=1,
    default=50,
    type=int,
    show_default=True,
    help="Set specific closure count per break. Use 0 to set no maximum. You cannot request exactly 0 closures.",
)
@click.option(
    "-l",
    "--insertion-length-per-closure",
    "insertion_length_per_closure",
    nargs=2,
    default=[1, 20],
    type=int,
    show_default=True,
    help="Optionally limit the minimum or maximum size of loops. Specify both min and max. If min and max are the same, only closures of a single size will be output. The script will accept non-sensical values and return everything it finds in your range, though this may be nothing for some ranges.",
)
@click.option(
    "-t",
    "--rmsd-threshold",
    "rmsd_threshold",
    nargs=1,
    default=0.25,
    type=float,
    show_default=True,
    help="Default minimum rmsd is 0.25, if you want to allow lower res, or restrict to higher res, specify a value.",
)
@click.option(
    "-s",
    "--silent-mode",
    "silent_mode",
    is_flag=True,
    help="This flag allows you to pass silent file(s) and get silent files back. Ignore if you would rather work with pdbs",
)
@click.option(
    "-m",
    "--max_table-depth",
    "max_tables",
    default=25,
    show_default=True,
    help="Limit the number of hashmaps to traverse. Useful only if you know how the hdf5 is structured and you want to do something special. Messing with this can drastically lengthen or worsen your run.",
)
@click.option(
    "-i",
    "--allowed-trim-depth",
    "allowed_trim_depth",
    default=0,
    show_default=True,
    help="Allowed depth to check at each chain_break. This essentially 'trims back' the structure to get more sampling endpoints",
)
def main(
    input_structure_paths,
    rosetta_flags_file="",
    loop_count_per_closure=50,
    insertion_length_per_closure=[1, 20],
    rmsd_threshold=0.25,
    silent_mode=False,
    one_with_everything=False,
    max_tables=25,
    allowed_trim_depth=0,
):
    """
    """
    hdf5, sfd, silent_index, silent_out = preload(rosetta_flags_file)
    logger.debug("preload complete")
    sorted_ds_list = get_sorted_ds_list(hdf5)
    sorted_ds_list = sorted_ds_list[:max_tables]
    # sorted_ds_print_list = [
    #     (ds.attrs["cart_resl"], ds.attrs["ori_resl"]) for ds in sorted_ds_list
    # ]
    # logger.debug(f"sorted_ds_print_list: {sorted_ds_print_list}")

    all_xforms = np.empty((0, 4, 4))
    chains_from_to = np.empty((0, 2))
    res_indices = np.empty((0, 2))
    poses_mask = np.empty(0)
    # don't be fooled, the man is for manager, not some sort of dated superhero-like alias
    pose_mans = []

    for struct_num, target_pose in enumerate(
        poses_from_paths(*input_structure_paths, silent_mode=silent_mode)
    ):
        logger.debug("pose obtained")
        pose_mans.append(
            poseman(pose=target_pose, allowed_trim_depth=allowed_trim_depth)
        )
        pm = pose_mans[struct_num]

        n_chains = pm.pose.num_chains()
        this_pm_xforms, this_pm_chains_from_to, this_pm_res_indices = pm.get_all_closure_xforms(
            *list((i, i + 1) for i in range(1, n_chains))
        )
        n_xforms = this_pm_xforms.shape[0]
        all_xforms = np.concatenate((all_xforms, this_pm_xforms), axis=0)
        chains_from_to = np.concatenate(
            (chains_from_to, this_pm_chains_from_to), axis=0
        )
        res_indices = np.concatenate(
            (res_indices, this_pm_res_indices), axis=0
        )
        this_poses_mask = np.full(n_xforms, struct_num)
        poses_mask = np.concatenate(poses_mask, this_poses_mask)

    num_poses = len(pose_mans)

    loops = np.zeros(num_poses)

    for kv_ds in sorted_ds_list:
        unclosed_mask = loops < loop_count_per_closure
        unclosed = loops[unclosed_mask]
        logger.debug(f"unclosed: {unclosed}")
        if unclosed.shape[0] == 0:
            logger.debug(
                f"Desired number of outputs found, not scanning bigger tables"
            )
            break
        logger.debug("building hashmap from archive")
        logger.debug(f"kv_ds.dtype: {kv_ds.dtype}")

        xbin_cart = kv_ds.attrs["cart_resl"]
        xbin_ori = kv_ds.attrs["ori_resl"]
        binner = xb(ori_resl=xbin_ori, cart_resl=xbin_cart)
        xbin_keys = binner.get_bin_index(np.array(all_xforms))

        gp_vals, key_mask = get_closure_hits(xbin_keys, kv_ds)
        flat_key_mask = key_mask.flatten()  # not sure why this isn't flat?

        # this is legacy from when there was one check per pose:
        # It's best to use one array for everything to exploit the parallelism of the hashmap
        # and numpy array math speedup, but then we have to figure out which pose everything came from
        main_array_hit_indices = np.nonzero(flat_key_mask == True)[0]
        logger.debug(f"main_array_hit_indices: {main_array_hit_indices}")

        string_ds = retrieve_string_archive(hdf5, xbin_cart, xbin_ori)

        for pose_number in range(num_poses):
            # cut all arrays down to just pose of interest
            pose_num_mask = (poses_mask == pose_number).astype(np.bool)
            this_pose_hits_mask = (flat_key_mask[pose_num_mask]).astype(
                np.bool
            )
            this_pose_chains_from_to = chains_from_to[pose_num_mask]
            this_pose_res_indices = res_indices[pose_num_mask]

            vals_mask = (flat_key_mask[pose_num_mask]).astype(np.bool)

            this_pose_gp_vals = gp_vals[vals_mask]

            this_pm = pose_mans[pose_number]

            masked_chains_from_to = this_pose_chains_from_to[
                this_pose_hits_mask
            ]
            masked_this_pose_res_indices = this_pose_res_indices[
                this_pose_hits_mask
            ]
            # disgusting zip to iterate with objects instead of indexes
            for (
                (str_ds_start, str_ds_offset),
                (c1, c2),
                (res_i_1, res_i_2),
            ) in zip(
                this_pose_gp_vals,
                masked_chains_from_to,
                masked_this_pose_res_indices,
            ):
                tag_entries = string_ds[
                    str_ds_start : str_ds_start + str_ds_offset
                ]
                loop_strings = [
                    loop_bytes.decode("UTF-8") for loop_bytes in tag_entries
                ]
                this_pm.record_closures(c1, c2, res_i_1, res_i_2, loop_strings)
    for pm in pose_mans:
        pm.build_and_dump_closures(
            silent_index,
            silent_out,
            default_silent(),
            loop_count_per_closure=loop_count_per_closure,
            insertion_length_per_closure=insertion_length_per_closure,
            rmsd_threshold=rmsd_threshold,
            out_path=".",
            rechain=False,
            allow_incomplete=False,
        )

    # df.to_csv("closure_data.csv")
    hdf5.close()


if __name__ == "__main__":
    main()
