import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from itertools import product
from collections import namedtuple

import numpy as np

import npose_util as nu
import npose_util_pyrosetta as nup

from .hashed_loop import (
    sfd_tag_slice,
    align_and_get_rmsd,
    link_poses,
    trim_pose,
)


# Named tuple declaration - I think this type of scope is normal for this object
DataContainer = namedtuple(
    "ClosureDataContainer", "catalog_set closure_data_list"
)
ClosureData = namedtuple(
    "Closure", "archive_string res_i_start res_i_end cart_resl ori_resl"
)
LoopContainer = namedtuple("Loop", "pose closure_data rmsd")


class PoseManager(object):
    """
    Manages the structure to close
    """

    def __init__(self, pose=None, allowed_trim_depth=0):
        self.pose = pose
        self.allowed_trim_depth = allowed_trim_depth
        self._closure_hits = {}
        self.build_npose()

    def build_npose(self):
        """
        Helper func to rebuild npose if you change the pose on the fly

        Try not to do this...
        """
        self._npose = nup.npose_from_pose(self.pose)
        self._tpose = nu.tpose_from_npose(self._npose)
        self._itpose = nu.itpose_from_tpose(self._tpose)

    def get_closure_xforms(self, chain_from, chain_to):
        """
        For this pair of chains, return the xform from Cterm of from to N of to

        also returns from_to_indices for the actual res pair used to make the xform

        This func automatically corrects trims for short fragments, do not
        assume array length.
        """
        pose_end = self.pose.size()
        from_chain_end_index = self.pose.chain_end(chain_from)
        to_chain_start_index = self.pose.chain_begin(chain_to)
        trim_back_to = max(from_chain_end_index - self.allowed_trim_depth, 1)
        trim_forward_to = min(
            to_chain_start_index + self.allowed_trim_depth, pose_end
        )
        from_to_indices = np.array(
            list(
                product(
                    range(trim_back_to, from_chain_end_index + 1),
                    range(to_chain_start_index, trim_forward_to + 1),
                )
            ),
            dtype=np.int,
        )
        # 0-indexing for array accession
        from_to_npose_indices = from_to_indices - 1
        # These are non-boolean integer index masks
        from_mask = from_to_npose_indices[:, 0]
        to_mask = from_to_npose_indices[:, 1]
        xforms = self._itpose[from_mask, :, :] @ self._tpose[to_mask, :, :]
        return xforms, from_to_indices

    def get_all_closure_xforms(self, *chain_pairs):
        """
        Returns all the closure xforms, as well as other data

        Returns closure_xforms,chain_from_to, and offset_from_to

        chain from to is a list of lists where each entry is [chain_from,chain_to]
        offset_from_to is a list of lists where each entry is the diff b/w chain begin/end
        and the residue the xform was built from: [...,[0,5],[-1,5],...]
        """
        all_xforms_list = []
        index_mask_list = []
        chain_mask_list = []
        for (c1, c2) in chain_pairs:
            xforms, from_to_indices = self.get_closure_xforms(c1, c2)
            n_indices = from_to_indices.shape[0]
            c1_mask = np.full(n_indices, c1)
            c2_mask = np.full(n_indices, c2)
            chain_mask = np.concatenate(
                (c1_mask[np.newaxis, :], c2_mask[np.newaxis, :]), axis=0
            ).T
            chain_mask_list.extend(chain_mask)
            index_mask_list.extend(from_to_indices)
            all_xforms_list.extend(xforms)
        xforms = np.array(all_xforms_list)
        res_indices = np.array(index_mask_list)
        chain_indices = np.array(chain_mask_list)

        return xforms, chain_indices, res_indices

    def record_closures(
        self,
        chain_start,
        chain_end,
        res_i_start,
        res_i_end,
        loop_strings,
        cart_resl=None,
        ori_resl=None,
    ):
        """

        """
        chain_closure_key = (chain_start, chain_end)
        existing_data_container = self._closure_hits.get(chain_closure_key)
        if existing_data_container is None:
            catalog_set = set(
                (archive_string, res_i_start, res_i_end)
                for archive_string in loop_strings
            )
            archive_data = [
                ClosureData(
                    archive_string, res_i_start, res_i_end, cart_resl, ori_resl
                )
                for archive_string in loop_strings
            ]
            dc = DataContainer(catalog_set, archive_data)
            self._closure_hits[chain_closure_key] = dc
        else:
            catalog_set = existing_data_container.catalog_set
            closure_data_list = existing_data_container.closure_data_list
            archive_data, catalog_data = zip(
                *[
                    [
                        ClosureData(
                            archive_string,
                            res_i_start,
                            res_i_end,
                            cart_resl,
                            ori_resl,
                        ),
                        (archive_string, res_i_start, res_i_end),
                    ]
                    for archive_string in loop_strings
                    if not (archive_string, res_i_start, res_i_end)
                    in catalog_set
                ]
            )

            closure_data_list.append(archive_data)
            catalog_set.update(catalog_data)
            dc = DataContainer(catalog_set, closure_data_list)
            self._closure_hits[chain_closure_key] = dc

    def get_closure_list(self, c1, c2):
        """
        Returns the ClosureData list for the given c1,c2

        Returns empty list otherwise (not None!)
        Try not to edit this list in place, it is not a copy
        """
        chain_closure_key = (c1, c2)
        logger.debug(f"getting closures for: {chain_closure_key}")
        data_container = self._closure_hits.get(chain_closure_key)
        if data_container is None:
            # TODO handle incomplete closures gracefully
            # logger.debug("None found!")
            return []
        else:
            # logger.debug(
            #     f"returning closures: {data_container.closure_data_list}"
            # )
            return data_container.closure_data_list

    def build_and_dump_closures(
        self,
        s_index,
        s_out,
        silent_archive,
        loop_count_per_closure=50,
        insertion_length_per_closure=[1, 20],
        rmsd_threshold=0.25,
        out_path=".",
        rechain=False,
        allow_incomplete=False,
        score_manager=None,
        max_check_depth_per_closure_list=150,
    ):
        """
        Attempts to build loop closures for this pose, returns a closure report
        """
        n_chains = self.pose.num_chains()
        min_size, max_size = insertion_length_per_closure

        loop_main_dict = {}

        for c1 in range(1, n_chains):
            c2 = c1 + 1
            passing_loops = []
            loop_main_dict[(c1, c2)] = passing_loops

            # load and align loops, save rmsd thresh passing ones to dict with some metadata
            closure_list = self.get_closure_list(c1, c2)
            if max_check_depth_per_closure_list:
                closure_list = closure_list[:max_check_depth_per_closure_list]
            for closure in closure_list:
                loop_string = closure.archive_string
                tag, start, end = loop_string.split(":")
                start = int(start)
                end = int(end)
                insertion_size = end - start - 1
                if not (min_size <= insertion_size <= max_size):
                    # logger.debug(f"insertion size mismatch")
                    # logger.debug(f"{min_size} , {insertion_size} , {max_size}")
                    continue
                try:
                    loop_pose = sfd_tag_slice(
                        s_index, s_out, silent_archive, tag, start, end
                    )
                except AssertionError:
                    continue

                bb_rmsd = align_and_get_rmsd(
                    loop_pose,
                    self.pose,
                    closure.res_i_start,
                    closure.res_i_end,
                )
                if not score_manager is None:
                    score_manager.add_score_data(
                        bb_rmsd, self.pose, closure, c1, c2
                    )
                if bb_rmsd > rmsd_threshold:
                    # logger.debug(
                    #     f"rmsd_threshold not met: {bb_rmsd} > {rmsd_threshold}"
                    # )
                    continue
                # remove overlap residues before anyone notices
                trim_pose(loop_pose, 2, loop_pose.size() - 1)

                # Save this structure to container for later
                passing_loops.append(
                    LoopContainer(loop_pose, closure, bb_rmsd)
                )
                if len(passing_loops) >= loop_count_per_closure:
                    continue
                # loop_main_dict[(c1, c2)] = passing_loops

            passing_loops.sort(key=lambda lc: lc.rmsd)
            passing_loops = passing_loops[:loop_count_per_closure]
            loop_main_dict[(c1, c2)] = passing_loops
        if not (allow_incomplete) and not (all(loop_main_dict.values())):
            logger.debug("not all loops closed!")
            logger.debug(f"loop_main_dict.items(): {loop_main_dict.items()}")
            return
        logger.debug("All loops closed!")
        logger.debug(f"loop_main_dict.items(): {loop_main_dict.items()}")

        # Careful, assumes you're not doing circular permutations
        sorted_chain_keys, loop_vals = zip(
            *sorted(loop_main_dict.items(), key=lambda item: item[0][0])
        )
        for output_n, val_set in enumerate(product(*loop_vals), 1):
            chain_set = set(
                chain for pair in sorted_chain_keys for chain in pair
            )
            chain_list = sorted(list(chain_set))
            # dumb container for keeping track of what's cut where
            # [first res, last res, loop pose]
            mod_dict = {c: [None, None, None] for c in chain_list}
            for (c1, c2), lc in zip(sorted_chain_keys, val_set):
                mod_dict[c1][1] = lc.closure_data.res_i_start
                mod_dict[c2][0] = lc.closure_data.res_i_end
                mod_dict[c1][2] = lc.pose
            final_pose_list = []
            for chain in chain_list:
                pose_clone = self.pose.clone()
                begin = mod_dict[chain][0]
                end = mod_dict[chain][1]
                loop = mod_dict[chain][2]
                trim_pose(pose_clone, begin, end)
                final_pose_list.append(pose_clone)
                if not (loop is None):
                    final_pose_list.append(loop)
            linked = link_poses(*final_pose_list, rechain=rechain)

            reloop_name = f"""{
                self.pose.pdb_info().name().split(".pdb")[0]
                }_{output_n}.pdb"""
            out_file = f"{out_path}/{reloop_name}"
            linked.dump_pdb(out_file)
