# hashed_loop

This is a collections of scripts for closing loops in pdb files (or silent files) as an add-on for the Rosetta protein suite.

## Installation Instructions:

### If you have a prebuilt table:
<br/>
<code> python -m pip install git+https://github.com/dmitropher/hashed_loop.git </code><br/>
<code> import_default_loop_table /path/to/loop.hf5 </code><br/>
(Tables are typically large, the copy operation can take a long time. The table is copied to the package resource directory rather than referenced.)

### Otherwise:
<br/>
<code> python -m pip install git+https://github.com/dmitropher/hashed_loop.git </code>


## Running:

The loop closer should be usable by running:
<br/><code>close_loop /path/to/my.pdb</code><br/>
Currently it only supports closing chain A to chain B

You can use <code>build_hash_loop_table --help<code/> to build your table. There is currently no default or best practice for this procedure until a global benchmark is complete. Email d.zorine@gmail.com for guidelines on making your own archive. This also typically takes a long time and a lot of RAM. This table can be set as default using the 

A link to a prebuilt table archive will be included in future releases
