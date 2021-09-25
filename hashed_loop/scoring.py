import pandas as pd


class ScoreManager(object):
    """
    Convenience object for holding and reporting information about the closure run
    """

    def __init__(self):
        self._main_df = None
        self._data_dict_list = []
        self.__rmsd_name = "closure_rmsd"
        self.__pdb_name = "pdb_name"
        self.__archive_string = "archive_string"
        self.__res_i_start = "pose_closure_res_start"
        self.__res_i_end = "pose_closure_res_end"
        self.__cart_resl = "cart_resl"
        self.__ori_resl = "ori_resl"

    def add_score_data(
        self, bb_rmsd, pose, closure_data, chain_from, chain_to
    ):
        """
        Add score data about a closure attempt
        """
        d = {
            self.__rmsd_name: bb_rmsd,
            self.__pdb_name: pose.pdb_info().name().split(".pdb")[0],
            self.__archive_string: closure_data.archive_string,
            self.__res_i_start: closure_data.res_i_start,
            self.__res_i_end: closure_data.res_i_end,
            self.__cart_resl: closure_data.cart_resl,
            self.__ori_resl: closure_data.ori_resl,
        }
        self._data_dict_list.append(d)

    def build_df_from_data(self):
        self._main_df = pd.DataFrame.from_dict(self._data_dict_list)

    def to_csv(self, path):
        """
        Builds and outputs the
        """
        self._main_df.to_csv(path)
