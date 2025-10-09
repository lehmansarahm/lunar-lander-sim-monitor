from collections import namedtuple
from enum import Enum
import numpy as np
import sys


"""
    ----------------------------------------------------------------------
    Store the lander's state as a named tuple of 8 values:
    ----------------------------------------------------------------------
        (x, y) = float coordinates in the 2D plane
        (x-dot, y-dot) = float linear velocities
        (theta) = float angle of orientation
        (theta-dot) = float angular velocity
        (l, r) = boolean flags indicating if the left, right legs are
            in contact with the ground
    ----------------------------------------------------------------------
"""
State = namedtuple("State",
                   field_names=["x", "y", "x_dot", "y_dot", "theta",
                                "theta_dot", "l", "r"])


"""
    ----------------------------------------------------------------------
    Store a lander experience as a named tuple of 6 values:
    ----------------------------------------------------------------------
        (state) = current state of the environment (itself an 8-value tuple)
        (action) = action to take as selected by the agent model
        (reward) = reward earned for taking the indicated action
        (next_state) = new state that the environment is in as a result of
            taking the indicated action
        (gt_action) = "ground truth" action, that is, what the generator
            model predicted should happen for the current state
        (done) = boolean flag indicating if we've reached a terminal state
    ----------------------------------------------------------------------
"""
Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward",
                                     "next_state", "gt_action", "done"])


class LanderAction(Enum):
    """
        ------------------------------------------------------------------
        Enum of possible actions that the lunar lander can take
        ------------------------------------------------------------------
    """

    DO_NOTHING = 0
    FIRE_LEFT = 1
    FIRE_MAIN = 2
    FIRE_RIGHT = 3

# ===== end enum LanderAction() ===================================================================


class DataGenerationType(Enum):
    """
        ------------------------------------------------------------------
        Enum of possible methods to generate new state data:
        ------------------------------------------------------------------
            (RANDOM) = completely randomly generated for all fields
            (RANDOM_BOUNDED) = randomly generated within the bounds of
                normal operations for each field
        ------------------------------------------------------------------
    """

    RANDOM = "randomly_generated"
    RANDOM_BOUNDED = "randomly_generated_with_bounds"

# ===== end enum DataGenerationType() =============================================================


class DataGenerator:


    @staticmethod
    def __validate_bounds(state_prop):
        assert type(state_prop) == list
        assert len(state_prop) == 2
    # ----- end function definition __validate_bounds() -------------------------------------------


    def __init__(self, bounds: State = None):
        if bounds is None:
            # ------------------------------------------------------------
            # Use the default bounds from the Lunar Lander environment
            # ------------------------------------------------------------
            #   import gymnasium as gym
            #   env = gym.make('LunarLander-v3', render_mode='rgb_array')
            #   print("LOW", env.observation_space.low)
            #   print("HIGH", env.observation_space.high)
            # ------------------------------------------------------------

            self.STATE_BOUNDS = State(
                x=(-2.5, 2.5),
                y=(-2.5, 2.5),
                x_dot=(-10.0, 10.0),
                y_dot=(-10.0, 10.0),
                theta=(-6.2831855, 6.2831855),
                theta_dot=(-10.0, 10.0),
                l=False, r=False
            )
        else:
            # don't need to validate "l" and "r" because they're just booleans
            self.__validate_bounds(self.STATE_BOUNDS.x)
            self.__validate_bounds(self.STATE_BOUNDS.y)
            self.__validate_bounds(self.STATE_BOUNDS.x_dot)
            self.__validate_bounds(self.STATE_BOUNDS.y_dot)
            self.__validate_bounds(self.STATE_BOUNDS.theta)
            self.__validate_bounds(self.STATE_BOUNDS.theta_dot)
            self.STATE_BOUNDS = bounds
        # end if-else block
    # ----- end function definition __init__() ----------------------------------------------------


    def generate_new(self, data_gen_type, gen_count):
        match data_gen_type.value:
            case DataGenerationType.RANDOM.value:
                rand_floats = np.random.rand(gen_count, 6)
                rand_bools = np.random.choice([True, False], size=(gen_count, 2))
                new_states = np.concatenate((rand_floats, rand_bools), axis=1)

            case DataGenerationType.RANDOM_BOUNDED.value:
                rand_x = np.random.uniform(self.STATE_BOUNDS.x[0],
                                           self.STATE_BOUNDS.x[1] + sys.float_info.min,
                                           size=gen_count)
                rand_y = np.random.uniform(self.STATE_BOUNDS.y[0],
                                           self.STATE_BOUNDS.y[1] + sys.float_info.min,
                                           size=gen_count)
                rand_x_dot = np.random.uniform(self.STATE_BOUNDS.x_dot[0],
                                               self.STATE_BOUNDS.x_dot[1] + sys.float_info.min,
                                               size=gen_count)
                rand_y_dot = np.random.uniform(self.STATE_BOUNDS.y_dot[0],
                                               self.STATE_BOUNDS.y_dot[1] + sys.float_info.min,
                                               size=gen_count)
                rand_theta = np.random.uniform(self.STATE_BOUNDS.theta[0],
                                               self.STATE_BOUNDS.theta[1] + sys.float_info.min,
                                               size=gen_count)
                rand_theta_dot = np.random.uniform(self.STATE_BOUNDS.theta_dot[0],
                                               self.STATE_BOUNDS.theta_dot[1] + sys.float_info.min,
                                               size=gen_count)
                rand_bools = np.random.choice([True, False], size=(gen_count, 2))
                new_states = np.column_stack((rand_x, rand_y, rand_x_dot, rand_y_dot,
                                              rand_theta, rand_theta_dot, rand_bools))

            case _:
                raise Exception("Data generation type:", data_gen_type, "is currently unsupported")
        # end match-block

        new_data = np.array([ State(*row) for row in new_states ])
        return new_data
    # ----- end function definition generate_new() ------------------------------------------------


# ===== end class DataGenerator() =================================================================

