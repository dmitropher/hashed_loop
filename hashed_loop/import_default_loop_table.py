import click
import os
from shutil import copyfile, move


from hashed_loop.file_io import (
    default_hdf5,
    default_silent,
    build_and_cache_gp_dicts,
)


# TODO Add link mode to this script so you don't ahve to copy resources every time
@click.command()
@click.argument("store_path")
@click.argument("silent_path")
@click.option("-l", "link_mode", is_flag=True, default=False)
def main(store_path, silent_path, link_mode=False):
    """
    Register the silent and data table, link_mode makes symlinks instead of copy

    """
    dest_path_store = default_hdf5()
    dest_path_silent = default_silent()

    os.makedirs(os.path.dirname(dest_path_store), exist_ok=True)
    os.makedirs(os.path.dirname(dest_path_silent), exist_ok=True)
    if link_mode:
        if os.path.isfile(dest_path_store):
            move(dest_path_store, dest_path_store + ".old")
        if os.path.isfile(dest_path_silent):
            move(dest_path_silent, dest_path_store + ".old")
        os.symlink(store_path, dest_path_store)
        os.symlink(silent_path, dest_path_silent)
    else:
        copyfile(store_path, dest_path_store)
        copyfile(silent_path, dest_path_silent)

    build_and_cache_gp_dicts(store_path)
