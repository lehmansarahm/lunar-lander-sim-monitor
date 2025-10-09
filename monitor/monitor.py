import gymnasium as gym
import imageio
import numpy as np
from sim.utils import DataGenerationType, DataGenerator
import tensorflow as tf


class Monitor:


    def extract_traces_and_preds(self, input_data=None, input_labels=None):
        """

        :param input_data:
        :param input_labels:
        :return:
        """

        input_data = self.training_data[0] if input_data is None else input_data
        input_labels = self.training_labels if input_labels is None else input_labels
        assert len(input_data) == len(input_labels)

        self.agent_q_values = self.agent_model.predict(input_data)
        np.save(self.OUTPUT_PRED_PATH, self.agent_q_values, allow_pickle=True)
        print("\nAgent model Q-values saved to file:", self.OUTPUT_PRED_PATH, "with shape:",
              self.agent_q_values.shape)

        monitor_traces = self.monitor_model.predict(input_data)
        self.monitor_trace_map = [ [] for _ in range(self.num_actions) ]
        for i in range(len(input_labels)):
            gt_action = int(input_labels[i])
            self.monitor_trace_map[gt_action].append(monitor_traces[i])
        # end record iteration loop

        for i in range(self.num_actions):
            action_traces = np.array(self.monitor_trace_map[i])
            np.save(self.OUTPUT_TRACE_PATH.format(i), action_traces, allow_pickle=True)
            print("Monitor traces for ground truth action:", i, "saved to file:",
                  self.OUTPUT_TRACE_PATH.format(i), "with shape:", action_traces.shape)
        # end action iteration loop
    # ----- end function definition extract_traces_and_preds() ------------------------------------


    def calculate_surprise_scores(self, metric_types):
        """

        :param metric_types:
        :return:
        """

        if self.monitor_trace_map is None:
            self.monitor_trace_map = [[] for _ in range(self.num_actions)]
            for action_num in range(self.num_actions):
                self.monitor_trace_map[action_num] = \
                    np.load(self.OUTPUT_TRACE_PATH.format(action_num), allow_pickle=True)
            # end for-loop
        # end if-block

        if self.agent_q_values is None:
            self.agent_q_values = np.load(self.OUTPUT_PRED_PATH, allow_pickle=True)
        # end if-block

        for metric_type in metric_types:
            metric_obj = metric_type(self.monitor_trace_map)
            # TODO - actually calculate the scores
            print(type(metric_obj))
        # end for-loop
    # ----- end function definition calculate_surprise_scores() -----------------------------------


    def generate_sample_video(self):
        """

        :return:
        """

        with imageio.get_writer(self.VIDEO_PATH, fps=self.VIDEO_FPS) as video:
            done = False
            state = self.monitor_env.reset()[0]
            frame = self.monitor_env.render()
            video.append_data(frame)

            while not done:
                state = np.expand_dims(state, axis=0)
                q_values = self.agent_model(state)
                action = np.argmax(q_values.numpy()[0])
                state, _, done, _, _ = self.monitor_env.step(action)
                frame = self.monitor_env.render()
                video.append_data(frame)
            # end while-loop
        # end with-block
    # ----- end function definition generate_sample_video() ---------------------------------------


    @staticmethod
    def generate_new_inputs(gen_types, gen_count=10_000):
        for gen_type in gen_types:
            output_filepath = "./input/{}.npy".format(gen_type.value)
            new_data = DataGenerator.generate_new(gen_type, gen_count)
            np.save(output_filepath, new_data, allow_pickle=True)
        # end for-loop
    # ----- end function definition generate_new_inputs() -----------------------------------------


    def __init__(self,
                 agent_model_path="./output/lunar_lander.keras",
                 training_data_path="./output/latest_buffer_states.npy",
                 training_label_path="./output/latest_buffer_gt_actions.npy",
                 test_data_path="./input/randomly_generated.npy",
                 output_trace_path="./output/latest_traces_{}.npy",
                 output_pred_path="./output/latest_preds.npy",
                 video_path="./output/lunar_lander.mp4",
                 video_fps=30):
        """

        :param agent_model_path:
        :param training_data_path:
        :param training_label_path:
        :param test_data_path:
        :param output_trace_path:
        :param output_pred_path:
        :param video_path:
        :param video_fps:
        """

        self.AGENT_MODEL_PATH = agent_model_path
        self.TRAINING_DATA_PATH = training_data_path
        self.TRAINING_LABEL_PATH = training_label_path
        self.TEST_DATA_PATH = test_data_path
        self.OUTPUT_TRACE_PATH = output_trace_path
        self.OUTPUT_PRED_PATH = output_pred_path
        self.VIDEO_PATH = video_path
        self.VIDEO_FPS = video_fps

        self.agent_model = tf.keras.models.load_model(self.AGENT_MODEL_PATH)
        self.num_actions = self.agent_model.get_layer("output").output.shape[1]

        monitor_layer = self.agent_model.get_layer("monitor")
        self.monitor_env = gym.make('LunarLander-v3', render_mode='rgb_array')
        self.monitor_model = tf.keras.models.Model(inputs=self.agent_model.inputs,
                                                   outputs=monitor_layer.output)

        self.training_data = np.load(self.TRAINING_DATA_PATH, allow_pickle=True)
        self.training_data = np.expand_dims(self.training_data, axis=0)

        self.training_labels = np.load(self.TRAINING_LABEL_PATH, allow_pickle=True)

        self.monitor_trace_map = None
        self.agent_q_values = None
    # ----- end function definition __init__() ----------------------------------------------------


    def __str__(self):
        output = [
            "LUNAR LANDER MONITOR PROPERTIES:",
            "",
            "\t Agent Model Path: \t\t\t" + self.AGENT_MODEL_PATH,
            "\t Agent Num Actions: \t\t" + str(self.num_actions),
            "",
            "\t Training Data Path: \t\t" + self.TRAINING_DATA_PATH,
            "\t Training Label Path: \t\t" + self.TRAINING_LABEL_PATH,
            "\t Test Data Path: \t\t\t" + self.TEST_DATA_PATH,
            "",
            "\t Output Trace Path: \t\t" + self.OUTPUT_TRACE_PATH,
            "\t OUtput Prediction Path: \t" + self.OUTPUT_PRED_PATH,
            "",
            "\t Output Video Path: \t\t" + self.VIDEO_PATH,
            "\t Output Video FPS: \t\t\t" + str(self.VIDEO_FPS),
        ]
        return "\n".join(output)
    # ----- end function definition __str__() ----------------------------------------------------


# ===== end class Monitor() =======================================================================
