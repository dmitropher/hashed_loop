#! /usr/bin/env python3
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import click
import getpy as gp
import numpy as np
from xbin import XformBinner as xb
import pandas as pd

from hashed_loop import (
    run_pyrosetta_with_flags,
    super_resi_by_bb,
    np_rt_from_residues,
    poses_from_silent,
    link_poses,
    sfd_tag_slice,
    get_chains,
    silent_preload,
    atom_coords,
    superposition_pose,
    subset_bb_rmsd,
)

from pyrosetta.rosetta.utility import vector1_bool as vector1_bool

# import silent_tools


def align_loop(loop_pose, target_pose, target_site):
    """
    Aligns loop in place to target at the site

    Alignment assumes target_site and target_site +1 are the flanking res!

    Make sure this numbering works out, or you're SOL
    """
    loop_pose_size = loop_pose.size()
    logger.debug(f"loop_pose_size: {loop_pose_size}")
    # super_resi_by_bb(loop_pose, target_pose, 1, target_site)
    init_coords = atom_coords(
        loop_pose,
        *[
            (resi, atom)
            for atom in ("N", "CA", "C")
            for resi in (1, loop_pose_size)
        ],
    )
    logger.debug(f"target_pose.size(): {target_pose.size()}")
    logger.debug(f"target_site: {target_site}")
    ref_coords = atom_coords(
        target_pose,
        *[
            (resi, atom)
            for atom in ("N", "CA", "C")
            for resi in (target_site, target_site + 1)
        ],
    )
    superposition_pose(loop_pose, init_coords, ref_coords)
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


def dump_report(
    closure_quality, attempted_poses, pose_mask, cart_resl, ori_resl
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
    df.to_csv("closure_data.csv")


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

    xbin_keys = binner.get_bin_index(np.array(all_xforms))
    key_mask = gp_dict.contains(xbin_keys)
    found_keys = xbin_keys[key_mask]
    # matching_poses = [
    #     pose for pose, is_found, in zip(target_poses, key_mask) if is_found
    # ]
    # logger.debug(matching_poses)
    closures_attempted = len(target_poses)
    # del target_poses

    pose_indices = np.nonzero(key_mask.flatten() == True)[0]
    logger.debug(pose_indices)
    gp_vals = gp_dict[found_keys].view(np.int32).reshape(-1, 2)
    # poses_closed = gp_vals.shape[0]
    closure_quality = np.full(closures_attempted, 1000.0)

    for gp_val, target_pose_i in zip(gp_vals, pose_indices):
        # logger.debug(gp_val)
        target_pose = target_poses[target_pose_i]
        chain_1, chain_2 = get_chains(target_pose)
        chain_a_end_index = chain_1.size()

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

            aligned_loop = align_loop(
                loop_pose, target_pose, chain_a_end_index
            )
            loop_pose_size = aligned_loop.size()

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
                logger.debug(f"better closure found: replacing")
                logger.debug(
                    f"closure_quality[{target_pose_i}]:{closure_quality[target_pose_i]} with {bb_rmsd}"
                )
                closure_quality[target_pose_i] = bb_rmsd

            aligned_loop.delete_residue_range_slow(
                loop_pose_size, loop_pose_size
            )
            aligned_loop.delete_residue_range_slow(1, 1)
            looped = link_poses(chain_1, aligned_loop, chain_2, rechain=True)
            reloop_name = f"""{
                target_pose.pdb_info().name().split(".pdb")[0]
                }_l{
                chain_a_end_index
                }_{loop_string}.pdb"""
            looped.dump_pdb(reloop_name)
        dump_report(
            closure_quality, target_poses, key_mask, xbin_cart, xbin_ori
        )


if __name__ == "__main__":
    main()
