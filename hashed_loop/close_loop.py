#! /usr/bin/env python3
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import click

# import getpy as gp
import numpy as np
from xbin import XformBinner as xb
import pandas as pd
import getpy as gp
import h5py

from hashed_loop import (
    run_pyrosetta_with_flags,
    poses_from_silent,
    silent_preload,
    get_closure_hits,
)
from hashed_loop.pose_manager import PoseManager as poseman
from hashed_loop.pose_manager import StructureManager
from hashed_loop.scoring import ScoreManager

from hashed_loop.file_io import (
    default_hdf5,
    default_silent,
    safe_load_pdbs,
    get_sorted_ds_list,
    cache_gp_dict,
    retrieve_gp_dict_from_cache,
    build_gp_dict,
)


def preload(rosetta_flags_file):
    """
    Util to preload all the stuff and initialize from user inputs
    """

    run_pyrosetta_with_flags(rosetta_flags_file)

    silent_index, silent_out = silent_preload(default_silent())
    hdf5_handle = h5py.File(default_hdf5(), "r")

    return hdf5_handle, silent_index, silent_out


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
@click.option(
    "-c",
    "--loop-count-per-closure",
    "loop_count_per_closure",
    nargs=1,
    default=1,
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
    default=0.35,
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
    default=10,
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
    loop_count_per_closure=1,
    insertion_length_per_closure=[1, 20],
    rmsd_threshold=0.35,
    silent_mode=False,
    one_with_everything=False,
    max_tables=10,
    allowed_trim_depth=0,
):
    """
    """
    hdf5, silent_index, silent_out = preload(rosetta_flags_file)
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
    # rosetta_xforms = np.empty((0, 4, 4))
    # don't be fooled, the man is for manager, not some sort of dated superhero-like alias
    pose_mans = []

    struct_manager = StructureManager(
        default_silent(), silent_out, silent_index
    )
    for struct_num, target_pose in enumerate(
        poses_from_paths(*input_structure_paths, silent_mode=silent_mode)
    ):
        logger.debug("pose obtained")
        pose_mans.append(
            poseman(
                pose=target_pose,
                allowed_trim_depth=allowed_trim_depth,
                structure_manager=struct_manager,
            )
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
        poses_mask = np.concatenate((poses_mask, this_poses_mask))
        # this_pose_rosetta_xforms = np.array(
        #     [
        #         np_rt_from_residues(
        #             target_pose.residue(r1), target_pose.residue(r2)
        #         )
        #         for (r1, r2) in this_pm_res_indices
        #     ]
        # )
        # rosetta_xforms = np.concatenate(
        #     (rosetta_xforms, this_pose_rosetta_xforms), axis=0
        # )

    num_poses = len(pose_mans)

    # loops = np.zeros(num_poses)
    for kv_ds in sorted_ds_list:
        # unclosed_mask = loops < loop_count_per_closure
        # unclosed = loops[unclosed_mask]
        # logger.debug(f"unclosed: {unclosed}")
        # if unclosed.shape[0] == 0:
        #     logger.debug(
        #         f"Desired number of outputs found, not scanning bigger tables"
        #     )
        #     break
        logger.debug("building hashmap from archive")

        xbin_cart = kv_ds.attrs["cart_resl"]
        xbin_ori = kv_ds.attrs["ori_resl"]
        binner = xb(ori_resl=xbin_ori, cart_resl=xbin_cart)
        xbin_keys = binner.get_bin_index(np.array(all_xforms))
        # r_xbin_keys = binner.get_bin_index(np.array(rosetta_xforms))

        # logger.debug(f"xbin_keys: {xbin_keys}")
        # logger.debug(f"r_xbin_keys: {r_xbin_keys}")

        key_type = np.dtype("i8")
        value_type = np.dtype("i8")
        gp_dict = retrieve_gp_dict_from_cache(
            xbin_ori, xbin_cart, key_type, value_type
        )
        if gp_dict is None:
            gp_dict = build_gp_dict(kv_ds, key_type, value_type)
            cache_gp_dict(gp_dict, xbin_ori, xbin_cart)

        gp_vals, key_mask = get_closure_hits(xbin_keys, gp_dict)
        logger.debug(f"gp_vals: {gp_vals}")
        logger.debug(f"key_mask: {key_mask}")
        flat_key_mask = key_mask.flatten()  # not sure why this isn't flat?

        # this is legacy from when there was one check per pose:
        # It's best to use one array for everything to exploit the parallelism of the hashmap
        # and numpy array math speedup, but then we have to figure out which pose everything came from
        main_array_hit_indices = np.nonzero(flat_key_mask == True)[0]
        logger.debug(f"main_array_hit_indices: {main_array_hit_indices}")

        string_ds = retrieve_string_archive(hdf5, xbin_cart, xbin_ori)

        for pose_number in range(num_poses):
            logger.debug(f"pose number: {pose_number}")
            # cut all arrays down to just pose of interest
            pose_num_mask = (poses_mask == pose_number).astype(np.bool)

            this_pose_hits_mask = (flat_key_mask[pose_num_mask]).astype(
                np.bool
            )
            n_hits = this_pose_hits_mask.sum()
            logger.debug(f"pose_num_mask: {pose_num_mask}")
            logger.debug(f"this_pose_hits_mask: {this_pose_hits_mask}")
            logger.debug(f"n_hits: {n_hits}")

            if n_hits == 0:
                continue

            poses_before_mask = (poses_mask < pose_number).astype(np.bool)
            hits_before_mask = (flat_key_mask[poses_before_mask]).astype(
                np.bool
            )
            n_hits_before = hits_before_mask.sum()

            this_pose_gp_vals = gp_vals[n_hits_before : n_hits_before + n_hits]

            this_pose_chains_from_to = chains_from_to[pose_num_mask]
            this_pose_res_indices = res_indices[pose_num_mask]

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
                this_pm.record_closures(
                    int(c1),
                    int(c2),
                    int(res_i_1),
                    int(res_i_2),
                    loop_strings,
                    cart_resl=xbin_cart,
                    ori_resl=xbin_ori,
                )
    scoreman = ScoreManager()
    for pm in pose_mans:

        for outpose in pm.build_closures(
            loop_count_per_closure=loop_count_per_closure,
            insertion_length_per_closure=insertion_length_per_closure,
            rmsd_threshold=rmsd_threshold,
            rechain=False,
            allow_incomplete=False,
            score_manager=scoreman,
            max_check_depth_per_closure_list = max(50,loop_count_per_closure),
        ):
            outpose.dump_pdb("." + "/" + outpose.pdb_info().name() + ".pdb")

        scoreman.to_csv("closure_data.csv")

    # df.to_csv("closure_data.csv")
    hdf5.close()


if __name__ == "__main__":
    main()
