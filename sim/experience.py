from enum import Enum
from sim.utils import DataGenerator


class Action(Enum):

    RANDOM = "randomly_generated"
    RANDOM_BOUNDED = "randomly_generated_with_bounds"

# ===== end enum DataGenerationType() =============================================================


class Experience:

    def __init__(self, curr_state, act_taken, imm_reward, next_state, gt_action, done):
        self.CURRENT_STATE = curr_state
        print("Hello World")