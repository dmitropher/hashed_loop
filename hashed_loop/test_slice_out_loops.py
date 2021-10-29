import click
import pyrosetta
import getpy as gp
import numpy as np
from xbin import XformBinner as xb

from itertools import groupby

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def read_file_lines(filename):
    with open(filename, "r") as myfile:
        lines = myfile.readlines()
    return lines


def read_flag_file(filename):
    """Reads the flag file, ignoring comments"""
    lines = read_file_lines(filename)
    # filter the lines
    lines = [l for l in lines if l.startswith("-")]
    return " ".join(lines)


def run_pyrosetta_with_flags(flags_file_path, mute=False):
    if not flags_file_path:
        pyrosetta.init("-mute all " if mute else "", silent=mute)
        return
    flags = read_flag_file(flags_file_path)
    flags_str = " ".join(flags.replace("\n", " ").split())
    pyrosetta.init(
        f"-mute all {flags_str}" if mute else flags_str, silent=mute
    )


def stub_from_residue(
    residue, center_atom="CA", atom1="N", atom2="CA", atom3="C"
):
    """
    Returns a stub. A wrapper for atom.xyz with the default of the bb atoms.
    """
    return pyrosetta.rosetta.core.kinematics.Stub(
        residue.atom(center_atom).xyz(),
        residue.atom(atom1).xyz(),
        residue.atom(atom2).xyz(),
        residue.atom(atom3).xyz(),
    )


def rt_from_residues(residue_1, residue_2):
    return pyrosetta.rosetta.core.kinematics.RT(
        stub_from_residue(residue_1), stub_from_residue(residue_2)
    )


def rotation_translation_to_np_array(rotation, translation):
    """
    Takes a rotation matrix and a translation vector and returns a h xform
    """
    return np.array(
        [
            [rotation.xx, rotation.xy, rotation.xz, translation.x],
            [rotation.yx, rotation.yy, rotation.yz, translation.y],
            [rotation.zx, rotation.zy, rotation.zz, translation.z],
            [0, 0, 0, 1],
        ]
    )


def np_rt_from_residues(residue_1, residue_2):
    rosetta_rt = rt_from_residues(residue_1, residue_2)
    return rotation_translation_to_np_array(
        rosetta_rt.get_rotation(), rosetta_rt.get_translation()
    )


def parse_loop_iter(pose):
    """
    returns loops from the pose with multiple start and endpoints
    """

    try:
        logging.debug("splitting chains and extracting dssp")
        chain_list = list(pose.split_by_chain())

        logging.debug("chains split")
        dssp_obj = pyrosetta.rosetta.core.scoring.dssp.Dssp(pose)
    except RuntimeError as e:
        print(e)
        print("unable to compute dssp, skipping")
        return
        yield
    logging.debug("dssp computed")
    # full_dssp_string = dssp_obj.compute(pose)
    full_dssp_string = dssp_obj.get_dssp_secstruct()

    logging.debug("dssp loaded")
    final_index = chain_list[0].size()
    dssp_chains = [full_dssp_string[:final_index]]
    for chain in chain_list[1:]:
        c_size = chain.size()
        start_index = final_index
        final_index += c_size
        dssp_chains.append(full_dssp_string[start_index:final_index])
    logging.debug("dssp extracted")

    for i, dssp_string in enumerate(dssp_chains, 1):
        for res_type_string, iterator in groupby(
            enumerate(dssp_string, 1), lambda x: x[1]
        ):
            for run in ([n for n, p in iterator],):
                # warning mixed 0/1 indexing here (zero indexing used for full_dssp_string)
                if res_type_string == "L":
                    begin = run[0] + pose.chain_begin(i) - 1
                    end = run[-1] + pose.chain_begin(i) + 1
                    if (
                        end < pose.size()
                        and begin > 0
                        and full_dssp_string[end - 1] != "L"
                        and full_dssp_string[begin - 1]
                    ):
                        logging.debug("returning begin and end")

                        yield begin, end


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


def link_poses(*poses, rechain=False):
    """
    returns a pose of any number of poses appended end to end

    Order of input poses must be correct N->C, rechain appends by jump and calls
    chains_from_termini on the final conformation and copies but does not mess
    with pdb_info beyond what append_pose_by_jump does.

    No alignment, fold tree not smoothed, can take a single pose, returns a copy
    """
    assert bool(len(poses)), "number of input poses must be greater than 0"
    # target = _pyrosetta.rosetta.core.pose.Pose()
    # target.detached_copy(poses[0])
    target = poses[0].clone()
    # n_jump = target.num_jump()
    assert bool(len(target.residues) > 0), "Cannot link poses with 0 residues!"
    if rechain:
        for i, pose in enumerate(poses[1:]):
            assert bool(
                len(pose.residues) > 0
            ), "Cannot link poses with 0 residues!"
            target.append_pose_by_jump(pose, 1)
        # target.conformation().chains_from_termini()
    else:
        if target.residues[-1].has_variant_type(
            pyrosetta.rosetta.core.chemical.UPPER_TERMINUS_VARIANT
        ) and (len(poses) - 1):
            pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue(
                target,
                pyrosetta.rosetta.core.chemical.UPPER_TERMINUS_VARIANT,
                len(target.residues),
            )
        for pose in poses[1:]:
            if target.residues[-1].has_variant_type(
                pyrosetta.rosetta.core.chemical.UPPER_TERMINUS_VARIANT
            ):
                pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue(
                    target,
                    pyrosetta.rosetta.core.chemical.UPPER_TERMINUS_VARIANT,
                    len(target.residues),
                )
            if pose.residues[1].has_variant_type(
                pyrosetta.rosetta.core.chemical.LOWER_TERMINUS_VARIANT
            ):
                pyrosetta.rosetta.core.pose.remove_variant_type_from_pose_residue(
                    pose,
                    pyrosetta.rosetta.core.chemical.LOWER_TERMINUS_VARIANT,
                    1,
                )
            pyrosetta.rosetta.core.pose.append_pose_to_pose(
                target, pose, False
            )
    return target


@click.command()
@click.argument("loop_struct_silent", nargs=1)
@click.option("-r", "--rosetta-flags", "rosetta_flags_file", default="")
def main(loop_struct_silent, rosetta_flags_file=""):
    """
    """

    run_pyrosetta_with_flags(rosetta_flags_file)
    sfd = pyrosetta.rosetta.core.io.silent.SilentFileData(
        pyrosetta.rosetta.core.io.silent.SilentFileOptions()
    )
    sfd.read_file(loop_struct_silent)

    silent_name = "loops_removed.silent"
    sfd_out = pyrosetta.rosetta.core.io.silent.SilentFileData(
        silent_name,
        False,
        False,
        "binary",
        pyrosetta.rosetta.core.io.silent.SilentFileOptions(),
    )
    for target_pose in poses_from_silent(loop_struct_silent):
        tag = target_pose.pdb_info().name()

        chain_a_copy = target_pose.clone().split_by_chain()[1]
        for begin, end in parse_loop_iter(chain_a_copy.clone()):
            pose_copy = chain_a_copy.clone()

            # pose_copy = target_pose.clone().split_by_chain()[1]
            loop_start = begin + 1
            loop_end = end - 1
            print(f"loop_start: {loop_start}, loop_end: {loop_end}")
            if loop_start <= 3:
                print(f"loop starts at: {loop_start}, this is too small")
                continue
            if loop_end >= pose_copy.size():
                print(
                    f"loop ends at: {loop_end}, this is the last residue, skipping"
                )
                continue
            print("editting pose to create chainbreak")
            try:
                # pose_copy.delete_residue_range_slow(loop_start, loop_end)
                copy_chain_a = pose_copy.clone()
                copy_chain_a.delete_residue_range_slow(
                    loop_start, copy_chain_a.size()
                )
                print("chain a complete")
                copy_chain_b = pose_copy.clone()
                copy_chain_b.delete_residue_range_slow(1, loop_end)
                print("chain b complete, linking")
            except RuntimeError as e:
                print("error encountered: ")
                print(e)
                print("skipping")
                continue
            linked_split = link_poses(copy_chain_a, copy_chain_b, rechain=True)
            pinf = pyrosetta.rosetta.core.pose.PDBInfo(linked_split)
            linked_split.pdb_info(pinf)
            struct = sfd_out.create_SilentStructOP()
            struct.fill_struct(linked_split, tag)
            sfd_out.add_structure(struct)
            sfd_out.write_silent_struct(struct, silent_name, False)

            logging.debug("struct written to silent")


if __name__ == "__main__":
    main()
