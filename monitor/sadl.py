from abc import ABC, abstractmethod


class SADL(ABC):


    @classmethod
    @abstractmethod
    def get_name(cls):
        pass
    # ---- end def get_name() ---------------------------------------------------------------------


    @abstractmethod
    def calculate(self, test_traces, pred_action) -> list[float]:
        """
        Placeholder function for each SADL to implement when calculating scores

        :param test_traces:
        :param pred_action:

        :return: scores
        """
        pass
    # ---- end def calculate() --------------------------------------------------------------------


    def __init__(self, trace_map):
        """

        :param trace_map:
        """

        self.TRAINING_TRACE_MAP = trace_map
        self.name = "sadl"
    # ---- end def __init__() ---------------------------------------------------------------------


# ===== end class SADL() ==========================================================================

