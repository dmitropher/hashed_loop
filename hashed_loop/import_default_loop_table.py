import click
import os
from shutil import copyfile


@click.command()
@click.argument("store_path")
@click.argument("silent_path")
def main(store_path, silent_path):
    """
    """
    dest_path_store = os.path.join(
        os.path.dirname(__file__), "resources/hdf5_archives/default.hf5"
    )
    os.makedirs(os.path.dirname(dest_path_store), exist_ok=True)
    copyfile(store_path, dest_path_store)

    dest_path_silent = os.path.join(
        os.path.dirname(__file__), "resources/silent_files/default.silent"
    )
    os.makedirs(os.path.dirname(dest_path_silent), exist_ok=True)
    copyfile(silent_path, dest_path_silent)
