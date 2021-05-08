#! /usr/bin/env python3
import logging
import math
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
    atom_coords,
    superposition_pose,
    subset_bb_rmsd,
    get_closure_hits,
)
from hashed_loop.file_io import (
    default_hdf5,
    default_silent,
    safe_load_pdbs,
    get_sorted_ds_list,
)

from pyrosetta.rosetta.utility import vector1_bool as vector1_bool

# import silent_tools


# def slice_and_align_loop(
#     loop_pose, loop_start, loop_end, target_pose, target_site
# ):
#     loop_pose_size = loop_pose.size()
#     sliced_loop = loop_pose.clone()
#     logger.debug(f"loop_pose_size: {loop_pose_size}")
#     logger.debug(f"loop_start: {loop_start}")
#     logger.debug(f"loop_end: {loop_end}")
#     # continue
#     if loop_end < int(loop_pose_size):
#         sliced_loop.delete_residue_range_slow(loop_end + 1, loop_pose_size)
#     if loop_start > 1:
#         sliced_loop.delete_residue_range_slow(1, loop_start - 1)
#
#     return align_loop(loop_pose, target_pose, target_site)


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
def main(
    input_structure_paths,
    rosetta_flags_file="",
    loop_count_per_closure=50,
    insertion_length_per_closure=[1, 20],
    rmsd_threshold=0.25,
    silent_mode=False,
    one_with_everything=False,
    max_tables=25,
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

    all_xforms = []
    target_poses = []

    for target_pose in poses_from_paths(
        *input_structure_paths, silent_mode=silent_mode
    ):
        logger.debug("pose obtained")

        chain_1, chain_2 = get_chains(target_pose, 1, 2)
        chain_a_end_index = chain_1.size()
        xform_to_close = np_rt_from_residues(
            chain_1.residues[chain_a_end_index], chain_2.residues[1]
        )
        all_xforms.append(xform_to_close)
        target_poses.append(target_pose)

    min_size = min(insertion_length_per_closure)
    max_size = max(insertion_length_per_closure)

    closures_attempted = len(target_poses)
    loops = np.zeros(closures_attempted)

    closure_quality = np.full(closures_attempted, 1000.0)
    closures_on_disk = [set() for i in range(closures_attempted)]

    df = pd.DataFrame()

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

        gp_keys, key_mask = get_closure_hits(xbin_keys, kv_ds)
        # matching_poses = [
        #     pose for pose, is_found, in zip(target_poses, key_mask) if is_found
        # ]
        # logger.debug(matching_poses)
        # del target_poses

        pose_indices = np.nonzero(key_mask.flatten() == True)[0]
        logger.debug(f"pose_indices: {pose_indices}")

        string_ds = retrieve_string_archive(hdf5, xbin_cart, xbin_ori)
        for gp_val, target_pose_i in zip(gp_vals, pose_indices):
            # logger.debug(gp_val)
            target_pose = target_poses[target_pose_i]
            chain_1, chain_2 = get_chains(target_pose, 1, 2)
            chain_a_end_index = chain_1.size()

            tag_entries = string_ds[gp_val[0] : gp_val[0] + gp_val[1]]

            # logger.debug(f"tag_entries: {tag_entries}")
            for loop_string_bytes in tag_entries:

                if loop_count_per_closure:
                    if loops[target_pose_i] >= loop_count_per_closure:
                        logger.debug(
                            f"loops_closed: {loops[target_pose_i] }, skipping this target:{target_pose_i}"
                        )
                        break
                loop_string = loop_string_bytes.decode("UTF-8")
                reloop_name = f"""{
                    target_pose.pdb_info().name().split(".pdb")[0]
                    }_l{
                    chain_a_end_index
                    }_{loop_string}.pdb"""
                if os.path.exists(reloop_name):
                    logger.debug(f"this closure is done already: skipping")
                    if reloop_name in closures_on_disk[target_pose_i]:
                        if loops[target_pose_i] == 0:
                            loops[target_pose_i] += 1
                    else:
                        closures_on_disk[target_pose_i].add(reloop_name)
                    continue
                # logger.debug(f"loop_string: {loop_string}")
                # working_target = chain_1.clone()
                # extract info from the archive
                tag, start, end = loop_string.split(":")
                start = int(start)
                end = int(end)

                # size of loop -2 ends which get chopped after alignment
                insertion_size = end - start - 1
                if not (min_size <= insertion_size <= max_size):
                    logger.debug(
                        f"insertion size ({insertion_size}) out of range: {insertion_length_per_closure}"
                    )
                    continue
                # get the loop from the silent, align, trim off ends
                # loop_pose = pose_from_sfd_tag(sfd, tag)
                # HACK careful, default_silent() is hardcoded here
                # TODO implement user ability to specify own tables for everything
                try:
                    loop_pose = sfd_tag_slice(
                        silent_index,
                        silent_out,
                        default_silent(),
                        tag,
                        start,
                        end + 1,
                    )
                except AssertionError as e:
                    logger.error(e)
                    logger.error("something went wrong in loop loading")
                    logger.error("passing")
                    continue

                aligned_loop = align_loop(
                    loop_pose, target_pose, chain_a_end_index
                )
                loop_pose_size = aligned_loop.size()
                # insertion_size = loop_pose_size - 2

                target_subset = vector1_bool(target_pose.size())
                aligned_loop_subset = vector1_bool(loop_pose_size)

                target_subset[chain_a_end_index] = True
                target_subset[chain_a_end_index + 1] = True
                aligned_loop_subset[1] = True
                aligned_loop_subset[loop_pose_size] = True

                bb_rmsd = subset_bb_rmsd(
                    target_pose,
                    aligned_loop,
                    target_subset,
                    aligned_loop_subset,
                    superimpose=False,
                )
                if bb_rmsd < closure_quality[target_pose_i]:
                    # logger.debug(f"better closure found: replacing")
                    # logger.debug(
                    #     f"closure_quality[{target_pose_i}]:{closure_quality[target_pose_i]} with {bb_rmsd}"
                    # )
                    closure_quality[target_pose_i] = bb_rmsd
                if bb_rmsd > rmsd_threshold:
                    # logger.debug(
                    #     f"bb_rmsd exceeds threshold {bb_rmsd} > {rmsd_threshold}"
                    # )
                    # logger.debug(f"Not building pose")
                    continue

                aligned_loop.delete_residue_range_slow(
                    loop_pose_size, loop_pose_size
                )
                aligned_loop.delete_residue_range_slow(1, 1)
                looped = link_poses(
                    chain_1, aligned_loop, chain_2, rechain=True
                )
                looped.dump_pdb(reloop_name)
                loops[target_pose_i] += 1
        df = update_report(
            df, closure_quality, target_poses, key_mask, xbin_cart, xbin_ori
        )
    df.to_csv("closure_data.csv")
    hdf5.close()


if __name__ == "__main__":
    main()
