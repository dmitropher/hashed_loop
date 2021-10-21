# hashed_loop

This is a collections of scripts for closing loops in pdb files (or silent files) as an add-on for the Rosetta protein suite.

<b> If something doesn't run, please don't hesitate to email me! (d.zorine@gmail.com) </b>

## Installation Instructions:

<br/>
You must have pyrosetta inside of the environment you install this package in. It is recommended that you install this inside of a conda environment where you have pyrosetta.


<br/>
<code> python -m pip install git+https://github.com/dmitropher/hashed_loop.git </code><br/>
<code> import_default_loop_table /path/to/loop.hf5 /path/to/loop.silent </code><br/>
(Tables are typically large, the copy operation can take a long time. The table is copied to the package resource directory rather than referenced.)

prebuilt data is available at:
https://files.ipd.uw.edu/dzorine/full.hf5
https://files.ipd.uw.edu/dzorine/default.silent


If you have access to IPD computational resources, contact me directly and I can send you the paths to these resources on the shared filesystem so that you don't have to maintain your own copy and use up disk space quota.


## Running:

The loop closer should be usable by running:
<br/><code>close_loop /path/to/my.pdb</code><br/>
Run the following for options and help
<br/><code>close_loop --help</code><br/>

## Bulding your own table (not recommended)
You can use: <br/> <code> build_hash_loop_table --help </code> <br/> for info on how to build your table. There is currently no default or best practice for this procedure until a global benchmark is complete. Email d.zorine@gmail.com for guidelines on making your own archive. This also typically takes a long time and a lot of RAM. This table can be set as default using the table install script above.
