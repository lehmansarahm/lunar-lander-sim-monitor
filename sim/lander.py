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


    def __get_experiences(self, memory_buffer):
        """
        Returns a random sample of experience tuples drawn from the memory buffer.

        Retrieves a random sample of experience tuples from the given memory_buffer and
        returns them as TensorFlow Tensors. The size of the random sample is determined by
        the mini-batch size (MINIBATCH_SIZE).

        Args:
            memory_buffer (deque):
                A deque containing experiences. The experiences are stored in the memory
                buffer as named tuples: namedtuple("Experience", field_names=["state",
                "action", "reward", "next_state", "done"]).

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

        experiences = random.sample(memory_buffer, k=self.MINIBATCH_SIZE)

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
    # ----- end function definition __get_experiences() -------------------------------------------


    def __execute_time_step(self, time_idx, state, replay_buffer, epsilon):
        """

        :param time_idx:
        :param state:
        :param replay_buffer:
        :param epsilon:
        :return:
        """

        # The goal for any given time step is to choose an action (A) for the current
        # state (S) using a greedy policy for the current value of epsilon

        # First, make sure 'state' is in the right shape for the Q-function network
        state_qn = np.expand_dims(state, axis=0)
        q_values = self.q_network(state_qn)

        # Next, pick an action with trade-off parameter epsilon.  With epsilon
        # probability, return a random action (explore).  Otherwise, return the
        # best option from the info we've collected so far (exploit).
        action = random.choice(np.arange(4)) if random.random() <= epsilon \
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
        replay_buffer.append(experience(state, action, reward, next_state, done))

        # Only update the network when the necessary conditions are met.
        update_step_reached = (time_idx + 1) % self.NUM_STEPS_FOR_UPDATE == 0
        minibatch_size_exceeded = len(replay_buffer) > self.MINIBATCH_SIZE
        update = update_step_reached and minibatch_size_exceeded

        if update:
            # Sample random mini-batch of experience tuples (S, A, R, S-prime)
            # from the replay buffer
            experiences = self.__get_experiences(replay_buffer)

            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            self.update_agent(experiences)
        # end if-block

        return next_state.copy(), reward, done
    # ----- end function definition __execute_time_step() -----------------------------------------


    def __execute_episode(self, ep_idx, replay_buffer, total_point_history, num_time_steps,
                          num_points, epsilon):
        """

        :param ep_idx:
        :param replay_buffer:
        :param total_point_history:
        :param num_time_steps:
        :param num_points:
        :param epsilon:
        :return:
        """

        # For each new episode, reset the environment to the default conditions and get the
        # corresponding starting state from which to begin this episode
        state = self.lander_env.reset()
        total_points = 0

        # Execute all time steps for this episode
        for t in range(num_time_steps):
            state, reward, done = self.__execute_time_step(t, state, replay_buffer, epsilon)
            total_points += reward
            if done:
                # If this time step lands us in a terminal condition, we can end this loop early
                break
            # end if-block
        # end time step for-loop

        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_points:])

        # Update the value of epsilon with our selected decay rate or the hard-coded minimum...
        # whichever is bigger
        new_epsilon = max(self.EPSILON_MIN, self.EPSILON_DECAY * epsilon)

        print(f"\rEpisode {ep_idx + 1} | Total point average of the last {num_points} " +
              "episodes: {av_latest_points:.2f}", end="")

        if (ep_idx + 1) % num_points == 0:
            print(f"\rEpisode {ep_idx + 1} | Total point average of the last {num_points} " +
                  "episodes: {av_latest_points:.2f}")
        # end if-block

        # We will consider that the environment is solved if we get an average of 200 points
        # in the last 100 episodes.
        if av_latest_points >= 200.0:
            print(f"\n\nEnvironment solved in {ep_idx + 1} episodes!")
            self.q_network.save('lunar_lander_model.h5')
            return None
        # end if-block

        return new_epsilon
    # ----- end function definition __execute_episode() -------------------------------------------


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
        states, actions, imm_rewards, next_states, done_vals = experiences

        # For each next state, calculate what the expected rewards would be for each possible
        # action, then pick the largest of those values to keep using "reduce_max"
        max_qsa = tf.reduce_max(self.gen_network(next_states), axis=-1)

        # Eventually, we will reach a termination point for this lander; each experience
        # record reflects this with the "done" flag.  If we have reached a terminal state,
        # use the immediate reward for that state (R(s)).  Otherwise, calculate the reward
        # using the full Bellman Equation (R(s) + γ max Q^(s',a'))
        bellman_rewards = [ r + (self.GAMMA * q) for r, q in zip(imm_rewards, max_qsa) ]
        y_targets = [ r if d else b for r, b, d in zip(imm_rewards, bellman_rewards, done_vals)]
        y_targets = tf.convert_to_tensor(y_targets)

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
        total_point_history = []

        # Initialize our replay buffer with the desired memory size
        replay_buffer = deque(maxlen=self.MEMORY_SIZE)

        # Set the generator network weights equal to the Q-Network weights to begin
        self.gen_network.set_weights(self.q_network.get_weights())

        # Execute the number of requested episodes
        for i in range(num_episodes):
            epsilon = self.__execute_episode(i, replay_buffer, total_point_history, num_time_steps,
                                             num_points, epsilon)
            if epsilon is None:
                # If epsilon is None, it means we solved the problem early!
                break
        # end episode for-loop

        total_time = time.time() - start
        print(f"\nTotal Runtime: {total_time:.2f} s ({(total_time / 60):.2f} min)")
    # ----- end function definition train_agent() -------------------------------------------------


    def __get_default_network(self):
        return Sequential([
            Input(shape=self.state_size),
            Dense(units=64, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=self.num_actions, activation='linear')
        ])
    # ----- end function definition __get_default_network() ---------------------------------------


    def __init__(self, buffer_size=100_000, steps_per_update=4, minibatch_size=64, alpha=1e-3,
                 epsilon_min=0.01, epsilon_decay=0.995, gamma=0.995, tau=1e-3):
        """
        Updates the weights of the generator and Q-function networks; denoted as a "tf.function"
        to allow TensorFlow to convert the logic to graph-format for improved performance

        Args:
            buffer_size: (int) size of the replay buffer
            steps_per_update: (int) number of time steps before we do an update
            alpha: (float) learning rate to use during training
            gamma: (float) discount / decay factor when calculating Q function
            tau: (float) soft update weighting parameter indicating how much we should favor
                the old network weights during updates

        """

        self.MEMORY_SIZE = buffer_size
        self.NUM_STEPS_FOR_UPDATE = steps_per_update
        self.MINIBATCH_SIZE = minibatch_size

        self.ALPHA = alpha
        self.EPSILON_MIN = epsilon_min
        self.EPSILON_DECAY = epsilon_decay
        self.GAMMA = gamma
        self.TAU = tau

        self.lander_env = gym.make('LunarLander-v3', render_mode='rgb_array')
        self.state_size = self.lander_env.observation_space.shape
        self.num_actions = self.lander_env.action_space.n

        # network to explore the state space and generate training data
        self.gen_network = self.__get_default_network()

        # network representing our learned action-value ("Q") function
        self.q_network = self.__get_default_network()

        # using "Adam" as our training optimizer for speed and performance
        self.optimizer = Adam(lr=self.ALPHA)
    # ----- end function definition __init__() ----------------------------------------------------


# ===== end class Lander() ========================================================================

