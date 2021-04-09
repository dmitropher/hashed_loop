import logging
from itertools import groupby


import os
import sys
import re

# from pyrosetta import *
# from pyrosetta.rosetta import *

# init("-in:file:silent_struct_type binary")

import distutils.spawn

sys.path.append(
    os.path.dirname(distutils.spawn.find_executable("silent_tools.py"))
)
import silent_tools

# from silent_tools import eprint


# maybe no hdf5 necessary
# import h5py
import numpy as np

import pyrosetta


def pose_from_sfd_tag(sfd, tag):

    ss = sfd.get_structure(tag)
    pose = pyrosetta.rosetta.core.pose.Pose()
    ss.fill_pose(pose)
    pose.pdb_info().name(tag)
    return pose


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
            pose = pose_from_sfd_tag(sfd, tag)
            yield pose
        except RuntimeError as e:
            print("issue loading pose, skipping")
            print(e)
            continue


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


def dssp_parse_loop_iter(pose):
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


def parse_loop_iter(pose, use_dssp=False, use_length=20):
    """
    returns fragments from the pose with multiple start and endpoints

    use_dssp returns only the fragments considered "loops", default behavior
    is to use all 3-20mers.

    dssp and n-mers are mutually exclusive

    """
    if use_dssp:
        logging.warning("Using dssp instead of n-mers")
        yield (dssp_parse_loop_iter(pose))
    else:
        for resn in range(1, pose.size() + 1 - use_length):
            for i in range(2, use_length + 1):
                yield resn, resn + i


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


def dumb_clean_pose(pose):
    """
    iterate through residues, remove anything that isn't protein
    """
    for i in range(pose.size(), 0, -1):
        prot = pose.residue(i).is_protein()
        canon = pose.residue(i).type().is_canonical_aa()
        if not (prot and canon):
            if pose.size() < 2:
                logging.debug("only one res left and it's not protein")
                raise RuntimeError(
                    "Pose only has one remaining res and it's not protein"
                )
            logging.debug(
                f"resdiue {i} is protein: {prot}, is canonical: {canon}, deleting"
            )
            pose.delete_residue_range_slow(i, i)
    logging.debug("non-protein res stripped, returning")
    return pose


def check_no(pose):
    """
    return false if any protein residue is missing N or O
    """
    logging.debug("checking residues for N and O")
    for i in range(1, pose.size() + 1):
        res = pose.residue(i)
        logging.debug(f"checking residues {i}")
        if not (res.is_protein()):
            logging.debug(f"res {i} is protein")
            # logging.debug(str(res))
            if not (res.has("N") or res.has("O")):
                return False
    return True


# Superposition Transform
def superposition_pose(mob_pose, init_coords, ref_coords):
    """
    Thin wrapper to fuse the creation of the rotation/translation and the apply
    """

    rotation = pyrosetta.rosetta.numeric.xyzMatrix_double_t()
    to_init_center = pyrosetta.rosetta.numeric.xyzVector_double_t()
    to_fit_center = pyrosetta.rosetta.numeric.xyzVector_double_t()
    pyrosetta.rosetta.protocols.toolbox.superposition_transform(
        init_coords, ref_coords, rotation, to_init_center, to_fit_center
    )
    pyrosetta.rosetta.protocols.toolbox.apply_superposition_transform(
        mob_pose, rotation, to_init_center, to_fit_center
    )
    return mob_pose


def super_resi_by_bb(mob_pose, targ_pose, mob_index, targ_index):
    init_coords = (
        pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
    )
    ref_coords = pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
    mob_res = mob_pose.residue(mob_index)
    targ_res = targ_pose.residue(targ_index)
    for atom in ("N", "CA", "C"):
        init_coords.append(mob_res.xyz(atom))
        ref_coords.append(targ_res.xyz(atom))
    superposition_pose(mob_pose, init_coords, ref_coords)
    return mob_pose


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


def sfd_tag_slice(silent_index, silent_out, silent_file, tag, start, end):
    """
    Sketchy read silent into stream, cut out seq desired by resnum

    bcov sorcery
    """
    with open(silent_file, errors="ignore") as sf:

        structure = silent_tools.get_silent_structure_file_open(
            sf, silent_index, tag
        )

        silent_out += structure[0]  # this is SCORE: by definition

        annotated_seq = None

        iline = 1
        found_it = False
        while iline < len(structure):
            line = structure[iline]
            if line.startswith("ANNOTATED_SEQUENCE:"):
                annotated_seq = line.split()[1]
            if line[0] in "ELH":
                found_it = True
                break
            iline += 1
        assert found_it
        assert not annotated_seq is None

        struct_res1 = iline

        my_seq = [
            x.group()
            for x in re.finditer("([A-Z]([[][^]]+[]])?)", annotated_seq)
        ]

        assert start > 0
        assert start <= len(my_seq)
        assert end > 0
        assert end <= len(my_seq)

        # debugging assert, remove this
        assert len(structure) - struct_res1 == len(my_seq)

        new_struct = structure[struct_res1 + start - 1 : struct_res1 + end - 1]
        new_seq = my_seq[start - 1 : end - 1]

        silent_out += (
            "ANNOTATED_SEQUENCE: " + "".join(new_seq) + " " + tag + "\n"
        )

        silent_out += "".join(new_struct)

        stream = pyrosetta.rosetta.std.istringstream(silent_out)
        sfd = pyrosetta.rosetta.core.io.silent.SilentFileData(
            pyrosetta.rosetta.core.io.silent.SilentFileOptions()
        )

        vec = pyrosetta.rosetta.utility.vector1_std_string()
        vec.append(tag)
        sfd.read_stream(stream, vec, True, "fake")

        pose = pyrosetta.rosetta.core.pose.Pose()
        sfd.get_structure(tag).fill_pose(pose)
        return pose


def silent_preload(silent_file_path):
    """
    Returns sfd, index
    """
    sfd = pyrosetta.rosetta.core.io.silent.SilentFileData(
        pyrosetta.rosetta.core.io.silent.SilentFileOptions()
    )
    sfd.read_file(silent_file_path)

    silent_index = silent_tools.get_silent_index(silent_file_path)

    silent_out = silent_tools.silent_header(silent_index)
    return sfd, silent_index, silent_out


def get_chains(pose):
    chains = pose.split_by_chain()
    chain_1 = chains[1]
    chain_2 = chains[2]
    return chain_1, chain_2
