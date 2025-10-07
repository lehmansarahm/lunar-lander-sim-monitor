
class Experience:

    # # Store experiences as named tuples
    # experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __init__(self):
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None
    # ----- end function definition __init__() ----------------------------------------------------

# ===== end class Experience() ====================================================================

