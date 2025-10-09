import numpy as np
from monitor.sadl import SADL
from scipy.stats import gaussian_kde
from tqdm import tqdm


class LSA(SADL):
    """
    Cite the source for this original approach
    """


    @staticmethod
    def __get_kde(act_traces):
        """Kernel density estimation

        Args:
            act_traces (list): List of activation traces in training set.

        Returns:
            kde (list): kde
            removed_cols (list): List of removed columns by variance threshold.
        """

        var_threshold = 1e-5
        removed_cols = []

        col_vectors = np.transpose(act_traces)
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < var_threshold and i not in removed_cols:
                removed_cols.append(i)
            # end if-statement
        # end for-loop

        refined_acts = np.transpose(act_traces)
        refined_acts = np.delete(refined_acts, removed_cols, axis=0)

        if refined_acts.shape[0] == 0:
            print("acts were removed by threshold {}".format(var_threshold))
        # end if-statement

        kde = gaussian_kde(refined_acts)
        return kde, removed_cols
    # ---- end def __get_kde() --------------------------------------------------------------------


    @staticmethod
    def __get_lsa(kde, act, removed_cols):
        """

        :param kde:
        :param act:
        :param removed_cols:

        :return:
        """

        refined_act = np.delete(act, removed_cols, axis=0)
        return np.ndarray.item(-kde.logpdf(np.transpose(refined_act)))
    # ---- end def __get_lsa() --------------------------------------------------------------------


    def calculate(self, test_traces, pred_action):
        """

        :param test_traces:
        :param pred_action:

        :return:
        """

        print("Calculating Likelihood-based Surprise Adequacy scores...")
        lsa_scores = []
        kde, removed_cols = self.kde_map[pred_action]

        for act in tqdm(test_traces):
            lsa_scores.append(self.__get_lsa(kde, act, removed_cols))
        # end for-loop

        return lsa_scores
    # ---- end def calculate_lsa() ----------------------------------------------------------------


    @classmethod
    def get_name(cls):
        return "lsa"
    # ---- end def get_name() ---------------------------------------------------------------------


    def __init__(self, trace_map):
        """

        :param trace_map:
        """

        super().__init__(trace_map)
        self.kde_map = [ [] for _ in range(len(trace_map)) ]
        for action_num in range(len(trace_map)):
            train_traces = trace_map[action_num]
            kde, removed_cols = self.__get_kde(train_traces)
            self.kde_map[action_num] = [ kde, removed_cols ]
        # end KDE iteration loop
    # ---- end def __init__() ---------------------------------------------------------------------


# ===== end class LSA() ===========================================================================

