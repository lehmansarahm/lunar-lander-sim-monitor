from abc import ABC, abstractmethod


class SADL(ABC):


    @abstractmethod
    def calculate(self, pred_action, test_traces, output_filepath):
        """
        Placeholder function for each SADL to implement when calculating scores

        :param pred_action:
        :param test_traces:
        :param output_filepath:

        :return: scores
        """
        pass
    # ---- end def calculate() --------------------------------------------------------------------


    def __init__(self, trace_map):
        """

        :param trace_map:
        """

        self.TRAINING_TRACE_MAP = trace_map
    # ---- end def __init__() ---------------------------------------------------------------------


# ===== end class SADL() ==========================================================================

