import pandas as pd


class ScoreManager(object):
    """
    Convenience object for holding and reporting information about the closure run
    """

    def __init__(self):
        self._main_df = None
