import math
import numpy as np
from monitor.sadl import SADL
import scipy as sp
from tqdm import tqdm


class MDSA(SADL):
    """
    Source:  Kim, Jinhan, et al. "Reducing dnn labelling cost using surprise adequacy: An
             industrial case study for autonomous driving." Proceedings of the 28th ACM Joint
             Meeting on European Software Engineering Conference and Symposium on the
             Foundations of Software Engineering. 2020.
    """


    def calculate(self, pred_action, test_traces, output_filepath):
        """
        Iterate through new predictions, calculate their distance from the training data
        using the authors' MDSA definition:  𝑀𝐷𝑆𝐴(𝑥) = √︃(𝛼 (𝑥)−𝜇 )𝑇𝑆−1(𝛼 (𝑥)−𝜇 )


        :param pred_action:
        :param test_traces:
        :param output_filepath:

        :return:
        """

        mdsa_scores = []
        for alpha in tqdm(test_traces):
            means_vector = np.array(self.means_vector_map[pred_action])
            inv_cov = np.array(self.inv_cov_map[pred_action])
            alpha_means_transpose = np.transpose(alpha - means_vector)
            mdsa = math.sqrt(np.dot(np.dot(alpha_means_transpose, inv_cov),
                                    (alpha - means_vector)))
            mdsa_scores.append(mdsa)
        # end for-loop

        np.save(output_filepath, mdsa_scores)
        return mdsa_scores
    # ---- end def calculate() --------------------------------------------------------------------


    def __init__(self, trace_map):
        """
        Calculate mean / INVERTED covariance matrix FOR EACH CLASS represented within the training
        activation map

        :param trace_map:
        """

        super().__init__(trace_map)

        self.means_vector_map = [ [] for _ in range(len(trace_map)) ]
        self.inv_cov_map = [ [] for _ in range(len(trace_map)) ]

        for action_num in range(len(trace_map)):
            train_traces = np.array(trace_map[action_num])
            self.means_vector_map[action_num] = train_traces.mean(axis=0)
            self.inv_cov_map[action_num] = sp.linalg.inv(np.cov(train_traces.T))
        # end for-loop
    # ---- end def __init__() ---------------------------------------------------------------------


# ===== end class MDSA() ==========================================================================
