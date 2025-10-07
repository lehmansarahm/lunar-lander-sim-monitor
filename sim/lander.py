from collections import deque, namedtuple
import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
import time

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


# Seed for the pseudo-random number generator.
SEED = 0

# Set the random seed for TensorFlow
tf.random.set_seed(SEED)

# Store experiences as named tuples
experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state", "done"])


class Lander:


    def __get_default_network(self):
        """

        :return:
        """

        return Sequential([
            Input(shape=self.state_size, name="input"),
            Dense(units=64, activation="relu", name="hidden1"),
            Dense(units=64, activation="relu", name="hidden2"),
            Dense(units=self.num_actions*2, activation="linear", name="monitor"),
            Dense(units=self.num_actions, activation="linear", name="output")
        ])
    # ----- end function definition __get_default_network() ---------------------------------------


    @staticmethod
    def __parse_experiences(experiences):
        """

        :param experiences:
        :return:
        """

        states = np.array([e.state for e in experiences if e is not None])
        states = tf.convert_to_tensor(states, dtype=tf.float32)

        actions = np.array([e.action for e in experiences if e is not None])
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        rewards = np.array([e.reward for e in experiences if e is not None])
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        next_states = np.array([e.next_state for e in experiences if e is not None])
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        done_flags = np.array([e.done for e in experiences if e is not None])
        done_flags = tf.convert_to_tensor(done_flags.astype(np.uint8), dtype=tf.float32)

        return states, actions, rewards, next_states, done_flags
    # ----- end function definition __parse_experiences() -----------------------------------------


    def __sample_experiences_from_replay_buffer(self):
        """
        Returns a random sample of experience tuples drawn from the memory buffer.

        Retrieves a random sample of experience tuples from the given memory_buffer and
        returns them as TensorFlow Tensors. The size of the random sample is determined by
        the mini-batch size (MINIBATCH_SIZE).

        Returns:
            A tuple (states, actions, rewards, next_states, done_vals) where:
                - states are the starting states of the agent.
                - actions are the actions taken by the agent from the starting states.
                - rewards are the rewards received by the agent after taking the actions.
                - next_states are the new states of the agent after taking the actions.
                - done_vals are the boolean values indicating if the episode ended.

            All tuple elements are TensorFlow Tensors whose shape is determined by the
            mini-batch size and the given Gym environment. For the Lunar Lander environment
            the states and next_states will have a shape of [MINIBATCH_SIZE, 8] while the
            actions, rewards, and done_vals will have a shape of [MINIBATCH_SIZE]. All
            TensorFlow Tensors have elements with dtype=tf.float32.
        """

        experiences = random.sample(self.replay_buffer, k=self.MINIBATCH_SIZE)
        return self.__parse_experiences(experiences)
    # ----- end function definition __get_experiences() -------------------------------------------


    def __save_replay_buffer(self):
        states, actions, rewards, next_states, done_flags = self.__parse_experiences(self.replay_buffer)
        buffer_components = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "done_flags": done_flags
        }

        output_path = "./output/latest_buffer_{}.npy"
        output_message = "Buffer component {} saved to file {} with shape {}"

        for key in buffer_components.keys():
            path = output_path.format(key)
            data = np.array(buffer_components[key])
            np.save(path, data, allow_pickle=True)
            print(output_message.format(key, path, data.shape))
        # end buffer component loop
    # ----- end function definition __save_replay_buffer() ----------------------------------------


    def compute_loss(self, experiences):
        """
        Calculates the loss for a new set of experience using Mean Squared Error (MSE).

        Args:
            experiences: (tuple) mini-batch of Experience records in the form of named tuples
                with properties - ["state", "action", "reward", "next_state", "done"]

        Returns:
            loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the MSE between the target
                values generated during exploration and the best action-value results
        """

        # Unpack the mini-batch of experience tuples
        states, actions, imm_rewards, next_states, done_flags = experiences

        # For each next state, calculate what the expected rewards would be for each possible
        # action, then pick the largest of those values to keep using "reduce_max"
        max_qsa = tf.reduce_max(self.gen_network(next_states), axis=-1)

        # Eventually, we will reach a termination point for this lander; each experience
        # record reflects this with the "done" flag.  If we have reached a terminal state,
        # use the immediate reward for that state (R(s)).  Otherwise, calculate the reward
        # using the full Bellman Equation (R(s) + γ max Q^(s',a'))
        y_targets = tf.convert_to_tensor(imm_rewards + (self.GAMMA * max_qsa * (1 - done_flags)))

        # Get the q_values from our trained action-value network and reshape to match y_targets
        q_values = self.q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))

        # Compute and return the loss
        loss = MSE(y_targets, q_values)
        return loss
    # ----- end function definition compute_loss() ------------------------------------------------


    @tf.function
    def update_agent(self, experiences):
        """
        Updates the weights of the generator and Q-function networks; denoted as a "tf.function"
        to allow TensorFlow to convert the logic to graph-format for improved performance

        Args:
            experiences: (tuple) mini-batch of Experience records in the form of named tuples
                with properties - ["state", "action", "reward", "next_state", "done"]

        """

        # Calculate the loss using Tensorflow's GradientTape for efficiency
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences)

        # Get the gradients of the loss with respect to the weights
        gradients = tape.gradient(loss, self.q_network.trainable_variables)

        # Update the weights of the Q-function network
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # *soft* update the weights of generator network based on Q-function network
        network_weights = zip(self.gen_network.weights, self.q_network.weights)
        for target_weights, q_net_weights in network_weights:
            target_weights.assign(self.TAU * q_net_weights + (1.0 - self.TAU) * target_weights)
    # ----- end function definition update_agent() ------------------------------------------------


    def execute_time_step(self, time_idx):
        """

        :param time_idx:

        :return:
        """

        # The goal for any given time step is to choose an action (A) for the current
        # state (S) using a greedy policy for the current value of epsilon

        # First, make sure 'state' is in the right shape for the Q-function network
        state_qn = np.expand_dims(self.current_state, axis=0)
        q_values = self.q_network(state_qn)

        # Next, pick an action with trade-off parameter epsilon.  With epsilon
        # probability, return a random action (explore).  Otherwise, return the
        # best option from the info we've collected so far (exploit).
        action = random.choice(np.arange(4)) if random.random() <= self.current_epsilon \
            else np.argmax(q_values.numpy()[0])

        # -----------------------------------------------------------------------
        # Take the action in the lander environment to receive the next state,
        # reward for this action, and a flag indicating whether we're done
        # -----------------------------------------------------------------------
        #   REMEMBER - "next_state" is a vector of 8 values:
        #       (x, y) = coordinates in the 2D plane
        #       (x-dot, y-dot) = linear velocities
        #       (theta) = angle of orientation
        #       (theta-dot) = angular velocity
        #       (l, r) = boolean flags indicating if the left, right legs are
        #           in contact with the ground
        # -----------------------------------------------------------------------
        next_state, reward, done, _, _ = self.lander_env.step(action)
        # -----------------------------------------------------------------------

        # Store experience tuple (S, A, R, S-prime) in the memory buffer.
        # We store the "done" variable as well for convenience.
        self.replay_buffer.append(experience(self.current_state, action, reward, next_state, done))

        # Only update the network when the necessary conditions are met.
        update_step_reached = (time_idx + 1) % self.NUM_STEPS_FOR_UPDATE == 0
        minibatch_size_exceeded = len(self.replay_buffer) > self.MINIBATCH_SIZE
        update = update_step_reached and minibatch_size_exceeded

        if update:
            # Sample random mini-batch of experience tuples (S, A, R, S-prime)
            # from the replay buffer
            experiences = self.__sample_experiences_from_replay_buffer()

            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            self.update_agent(experiences)
        # end if-block

        self.current_state = next_state.copy()
        return reward, done
    # ----- end function definition execute_time_step() -------------------------------------------


    def execute_episode(self, ep_idx, num_time_steps, num_points):
        """

        :param ep_idx:
        :param num_time_steps:
        :param num_points:

        :return:
        """

        # For each new episode, reset the environment to the default conditions and get the
        # corresponding starting state from which to begin this episode
        self.current_state = self.lander_env.reset()[0]
        total_points = 0

        # Execute all time steps for this episode
        for t in range(num_time_steps):
            reward, done = self.execute_time_step(t)
            total_points += reward
            if done:
                # If this time step lands us in a terminal condition, we can end this loop early
                break
            # end if-block
        # end time step for-loop

        self.total_point_history.append(total_points)
        avg_latest_points = np.mean(self.total_point_history[-num_points:])

        # Update the value of epsilon with our selected decay rate or the hard-coded minimum...
        # whichever is bigger
        self.current_epsilon = max(self.EPSILON_MIN, self.EPSILON_DECAY * self.current_epsilon)

        print(f"\rEpisode {ep_idx + 1} | Total point average of the last {num_points} " +
              f"episodes: {avg_latest_points:.2f}", end="")

        if (ep_idx + 1) % num_points == 0:
            print(f"\rEpisode {ep_idx + 1} | Total point average of the last {num_points} " +
                  f"episodes: {avg_latest_points:.2f}")
        # end if-block

        # We will consider that the environment is solved if we get an average of 200 points
        # in the last 100 episodes.
        if avg_latest_points >= self.TARGET_SCORE:
            print(f"\n\nEnvironment solved in {ep_idx + 1} episodes!")
            self.q_network.save("./output/lunar_lander_model.keras")
            self.__save_replay_buffer()
            return True
        else:
            return False
        # end if-block
    # ----- end function definition execute_episode() ---------------------------------------------


    def train_agent(self, num_episodes=2000, num_time_steps=1000, num_points=100, epsilon=1.0):
        """
        Trains the agent with an epsilon-greedy policy over a configurable number of episodes

        Args:
            num_episodes: (int) number of episodes for which to train
            num_time_steps: (int) number of time steps to run per episode
            num_points: (int) number of points to include when averaging
            epsilon: (float) exploration / exploitation trade-off parameter; initially favors
                exploration but will gradually shift toward exploitation as learning progresses

        """

        start = time.time()
        self.total_point_history = []
        self.current_epsilon = epsilon

        # Initialize our replay buffer with the desired memory size
        self.replay_buffer = deque(maxlen=self.MEMORY_SIZE)

        # Set the generator network weights equal to the Q-Network weights to begin
        self.gen_network.set_weights(self.q_network.get_weights())

        # Execute the number of requested episodes
        for i in range(num_episodes):
            if self.execute_episode(i, num_time_steps, num_points):
                # If episode execution returns "True", it means we solved the
                # problem early and don't need to do any more processing!
                break
            # end if-block
        # end episode for-loop

        total_time = time.time() - start
        print(f"\nTotal Runtime: {total_time:.2f} s ({(total_time / 60):.2f} min)")
    # ----- end function definition train_agent() -------------------------------------------------


    def __init__(self, buffer_size=100_000, steps_per_update=4, minibatch_size=64, target_score=200.0,
                 alpha=1e-3, gamma=0.995, tau=1e-3, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Updates the weights of the generator and Q-function networks; denoted as a "tf.function"
        to allow TensorFlow to convert the logic to graph-format for improved performance

        Args:
            buffer_size: (int) size of the replay buffer
            steps_per_update: (int) number of time steps before we do an update
            minibatch_size: (int) etc.
            target_score: (float) etc.
            alpha: (float) learning rate to use during training
            gamma: (float) discount / decay factor when calculating Q function
            tau: (float) soft update weighting parameter indicating how much we should favor
                the old network weights during updates
            epsilon_min: (float) etc.
            epsilon_decay: (float) etc.

        """

        self.MEMORY_SIZE = buffer_size
        self.NUM_STEPS_FOR_UPDATE = steps_per_update
        self.MINIBATCH_SIZE = minibatch_size
        self.TARGET_SCORE = target_score

        self.ALPHA = alpha
        self.GAMMA = gamma
        self.TAU = tau

        self.EPSILON_MIN = epsilon_min
        self.EPSILON_DECAY = epsilon_decay

        self.lander_env = gym.make('LunarLander-v3', render_mode='rgb_array')
        self.state_size = self.lander_env.observation_space.shape
        self.num_actions = self.lander_env.action_space.n

        # network to explore the state space and generate training data
        self.gen_network = self.__get_default_network()

        # network representing our learned action-value ("Q") function
        self.q_network = self.__get_default_network()

        # using "Adam" as our training optimizer for speed and performance
        self.optimizer = Adam(learning_rate=self.ALPHA)

        # some training properties that are useful to keep track of
        self.replay_buffer = None
        self.current_state = None
        self.current_epsilon = None
        self.total_point_history = None
    # ----- end function definition __init__() ----------------------------------------------------


    def __str__(self):
        output = [
            "LUNAR LANDER PROPERTIES:",
            "",
            "\t Replay Buffer Size: \t\t\t\t" + str(self.MEMORY_SIZE),
            "\t Steps Per Update: \t\t\t\t\t" + str(self.NUM_STEPS_FOR_UPDATE),
            "\t Mini-batch Size: \t\t\t\t\t" + str(self.MINIBATCH_SIZE),
            "\t Target Score to Finish: \t\t\t" + str(self.TARGET_SCORE),
            "",
            "\t Learning Rate (Alpha): \t\t\t" + str(self.ALPHA),
            "\t Q-func Discount Factor (Gamma): \t" + str(self.GAMMA),
            "\t Soft Update Trade-off Param (Tau): " + str(self.TAU),
            "",
            "\t Epsilon - Min Value: \t\t\t\t" + str(self.EPSILON_MIN),
            "\t Epsilon - Decay Rate: \t\t\t\t" + str(self.EPSILON_DECAY),
            "",
            "\t Env. Num Avail. Actions: \t\t\t" + str(self.num_actions),
            "\t Env. State Size: \t\t\t\t\t" + str(self.state_size)
        ]
        return "\n".join(output)
    # ----- end function definition __str__() ----------------------------------------------------


# ===== end class Lander() ========================================================================

