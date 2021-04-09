#! /usr/bin/env python3
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import click
import getpy as gp
import numpy as np
from xbin import XformBinner as xb

from hashed_loop import (
    run_pyrosetta_with_flags,
    super_resi_by_bb,
    np_rt_from_residues,
    poses_from_silent,
    link_poses,
    sfd_tag_slice,
    get_chains,
    silent_preload,
)


# import silent_tools


def align_loop(loop_pose, target_pose, target_site):
    """
    Aligns loop in place to target at the site, trims first/last res
    """
    loop_pose_size = loop_pose.size()
    logger.debug(f"loop_pose_size: {loop_pose_size}")
    super_resi_by_bb(loop_pose, target_pose, 1, target_site)
    loop_pose.delete_residue_range_slow(loop_pose_size, loop_pose_size)
    loop_pose.delete_residue_range_slow(1, 1)
    return loop_pose


def slice_and_align_loop(
    loop_pose, loop_start, loop_end, target_pose, target_site
):
    loop_pose_size = loop_pose.size()
    sliced_loop = loop_pose.clone()
    logger.debug(f"loop_pose_size: {loop_pose_size}")
    logger.debug(f"loop_start: {loop_start}")
    logger.debug(f"loop_end: {loop_end}")
    # continue
    if loop_end < int(loop_pose_size):
        sliced_loop.delete_residue_range_slow(loop_end + 1, loop_pose_size)
    if loop_start > 1:
        sliced_loop.delete_residue_range_slow(1, loop_start - 1)

    return align_loop(loop_pose, target_pose, target_site)


def preload(
    gp_dict_file, loop_archive, loop_struct_silent, rosetta_flags_file
):
    """
    Util to preload all the stuff and initialize from user inputs
    """
    key_type = np.int64
    value_type = np.int64
    gp_dict = gp.Dict(key_type, value_type)
    gp_dict.load(gp_dict_file)

    with open(loop_archive, "r") as f:
        loop_list = f.read().splitlines()
    run_pyrosetta_with_flags(rosetta_flags_file)

    sfd, silent_index, silent_out = silent_preload(loop_struct_silent)

    return gp_dict, loop_list, sfd, silent_index, silent_out


@click.command()
@click.argument("gp_dict_file", nargs=1)
@click.argument("loop_archive", nargs=1)
@click.argument("loop_struct_silent", nargs=1)
@click.argument("target_struct_silent", nargs=1)
@click.option(
    "-r",
    "--rosetta-flags",
    "rosetta_flags_file",
    default="",
    show_default=True,
)
@click.option(
    "-l",
    "--max-loops-per-closure",
    "max_loops",
    default=5,
    type=int,
    show_default=True,
)
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
def main(
    gp_dict_file,
    loop_archive,
    loop_struct_silent,
    target_struct_silent,
    rosetta_flags_file="",
    max_loops=5,
    xbin_cart=1.0,
    xbin_ori=15.0,
):
    """
    """
    gp_dict, loop_list, sfd, silent_index, silent_out = preload(
        gp_dict_file, loop_archive, loop_struct_silent, rosetta_flags_file
    )
    logger.debug("preload complete")

    binner = xb(ori_resl=xbin_ori, cart_resl=xbin_cart)

    all_xforms = []
    target_poses = []

    for target_pose in poses_from_silent(target_struct_silent):
        logger.debug("pose obtained")

        chain_1, chain_2 = get_chains(target_pose)
        chain_a_end_index = chain_1.size()
        xform_to_close = np_rt_from_residues(
            chain_1.residues[chain_a_end_index], chain_2.residues[1]
        )
        all_xforms.append(xform_to_close)
        target_poses.append(target_pose)

    xbin_keys = binner.get_bin_index(np.array([all_xforms]))
    key_mask = gp_dict.contains(xbin_keys)
    found_keys = xbin_keys[key_mask]
    matching_poses = np.array(target_poses)[key_mask]
    del target_poses
    gp_vals = gp_dict[found_keys]
    for gp_val in gp_vals:

        tag_entries = loop_list[gp_val[0] : gp_val[0] + gp_val[1]]
        loops = 1
        for loop_string in tag_entries:
            if max_loops:
                if loops > max_loops:
                    break
                else:
                    loops += 1
            # working_target = chain_1.clone()
            # extract info from the archive
            tag, start, end = loop_string.split(":")
            start = int(start)
            end = int(end)
            # get the loop from the silent, align, trim off ends
            # loop_pose = pose_from_sfd_tag(sfd, tag)
            loop_pose = sfd_tag_slice(
                silent_index,
                silent_out,
                loop_struct_silent,
                tag,
                start,
                end + 1,
            )

            sliced_loop = align_loop(loop_pose, chain_1, chain_a_end_index)

            looped = link_poses(chain_1, sliced_loop, chain_2, rechain=True)
            reloop_name = f"""{
                target_pose.pdb_info().name().split(".pdb")[0]
                }_l{
                chain_a_end_index
                }_{loop_string}.pdb"""
            looped.dump_pdb(reloop_name)


if __name__ == "__main__":
    main()
