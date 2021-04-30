import click
import os
from shutil import copyfile


@click.command()
@click.argument("store_path")
def main(store_path):
    """
    """
    dest_path = os.path.join(
        os.path.dirname(__file__), "resources/hdf5_archives/default.hf5"
    )
    os.makedirs(os.path.dirname(dest_path))
    copyfile(store_path, dest_path)
