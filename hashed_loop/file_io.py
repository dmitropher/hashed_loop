import os

import pyrosetta


def safe_load_pdbs(pdbs):
    for pdb in pdbs:
        try:
            yield pyrosetta.pose_from_pdb(pdb)
        except RuntimeError as e:
            print(e)
            print(f"unable to load: {pdb}")
            continue


def default_hdf5():
    """
    Return the hashed_loop default HDF5 path

    Do not mess with the HDF5 at this path unless you know what you are doing
    """
    hdf5_path = os.path.join(
        os.path.dirname(__file__), "resources/hdf5_archives/default.hf5"
    )
    return hdf5_path


def default_silent():
    """
    Return the hashed_loop default silent path

    Do not mess with the silent file at this path unless you know what you are doing
    """
    silent_path = os.path.join(
        os.path.dirname(__file__), "resources/silent_files/default.silent"
    )
    return silent_path
