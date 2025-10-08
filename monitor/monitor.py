import gymnasium as gym
import imageio
import numpy as np
import tensorflow as tf

from monitor.sadl import SADL


class Monitor:


    def extract_traces_and_preds(self, input_data=None):
        """

        :param input_data:
        :return:
        """

        input_data = self.training_data[0] if input_data is None else input_data

        self.agent_q_values = self.q_func_model.predict(input_data)
        agent_pred_actions = np.argmax(self.agent_q_values, axis=1)
        np.save(self.PRED_PATH, self.agent_q_values, allow_pickle=True)
        print("\nAgent model Q-values saved to file:", self.PRED_PATH, "with shape:",
              self.agent_q_values.shape)

        monitor_traces = self.monitor_model.predict(input_data)
        self.monitor_trace_map = [ [] for _ in range(self.num_actions) ]
        for i in range(len(agent_pred_actions)):
            pred_action = agent_pred_actions[i]
            self.monitor_trace_map[pred_action].append(monitor_traces[i])
        # end record iteration loop

        for i in range(self.num_actions):
            action_traces = np.array(self.monitor_trace_map[i])
            np.save(self.TRACE_PATH.format(i), action_traces, allow_pickle=True)
            print("Monitor traces for predicted action:", i, "saved to file:",
                  self.TRACE_PATH.format(i), "with shape:", action_traces.shape)
        # end action iteration loop
    # ----- end function definition extract_traces_and_preds() ------------------------------------


    def calculate_surprise_scores(self, metric_types):
        if self.monitor_trace_map is None:
            self.monitor_trace_map = [[] for _ in range(self.num_actions)]
            for action_num in range(self.num_actions):
                self.monitor_trace_map[action_num] = \
                    np.load(self.TRACE_PATH.format(action_num), allow_pickle=True)
            # end for-loop
        # end if-block

        if self.agent_q_values is None:
            self.agent_q_values = np.load(self.PRED_PATH, allow_pickle=True)
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
                q_values = self.q_func_model(state)
                action = np.argmax(q_values.numpy()[0])
                state, _, done, _, _ = self.monitor_env.step(action)
                frame = self.monitor_env.render()
                video.append_data(frame)
            # end while-loop
        # end with-block
    # ----- end function definition generate_sample_video() ---------------------------------------


    def __init__(self,
                 model_path="./output/lunar_lander.keras",
                 data_path="./output/latest_buffer_states.npy",
                 video_path="./output/lunar_lander.mp4", video_fps=30,
                 trace_path="./output/latest_traces_{}.npy",
                 pred_path="./output/latest_preds.npy"):
        """

        :param model_path:
        :param data_path:
        :param video_path:
        :param video_fps:
        :param trace_path:
        :param pred_path:
        """

        self.MODEL_PATH = model_path
        self.TRAINING_DATA_PATH = data_path
        self.VIDEO_PATH = video_path
        self.VIDEO_FPS = video_fps
        self.TRACE_PATH = trace_path
        self.PRED_PATH = pred_path

        self.q_func_model = tf.keras.models.load_model(self.MODEL_PATH)
        self.num_actions = self.q_func_model.get_layer("output").output.shape[1]

        monitor_layer = self.q_func_model.get_layer("monitor")
        self.monitor_env = gym.make('LunarLander-v3', render_mode='rgb_array')
        self.monitor_model = tf.keras.models.Model(inputs=self.q_func_model.inputs,
                                                   outputs=monitor_layer.output)

        self.training_data = np.load(self.TRAINING_DATA_PATH, allow_pickle=True)
        self.training_data = np.expand_dims(self.training_data, axis=0)

        self.monitor_trace_map = None
        self.agent_q_values = None
    # ----- end function definition __init__() ----------------------------------------------------


    def __str__(self):
        output = [
            "LUNAR LANDER MONITOR PROPERTIES:",
            "",
            "\t Q-func Model Path: \t\t" + self.MODEL_PATH,
            "\t Q-Model Num Actions: \t\t" + str(self.num_actions),
            "",
            "\t Training Data Path: \t\t" + self.TRAINING_DATA_PATH,
            "\t Output Video Path: \t\t" + self.VIDEO_PATH,
            "\t Output Video FPS: \t\t\t" + str(self.VIDEO_FPS),
            "",
            "\t Monitor Trace Path: \t\t" + self.TRACE_PATH,
            "\t Model Prediction Path: \t" + self.PRED_PATH
        ]
        return "\n".join(output)
    # ----- end function definition __str__() ----------------------------------------------------


# ===== end class Monitor() =======================================================================
