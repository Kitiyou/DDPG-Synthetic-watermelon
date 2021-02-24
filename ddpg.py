import os.path
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from environment import Environment

RESULT_FILE = 'result.txt'

ACTOR_LR = 0.0005  # 动作网络学习率
CRITIC_LR = 0.0005  # 价值网络学习率
TAU = 0.01  # 目标网络更新速率
GAMMA = 0.98  # 奖励衰减因子

EPISODE_NUM = 400


class DDPGAgent:

    def __init__(self, state_shape, load_path=None):
        # 定义网络
        self.actor = self._build_actor(state_shape)
        self.critic = self._build_critic(state_shape)
        self.target_actor = self._build_actor(state_shape)
        self.target_critic = self._build_critic(state_shape)
        if load_path:
            self.actor.load_weights(os.path.join(load_path, 'actor'))
            self.critic.load_weights(os.path.join(load_path, 'critic'))
            self.target_actor.load_weights(os.path.join(load_path, 'target_actor'))
            self.target_critic.load_weights(os.path.join(load_path, 'target_critic'))
        else:
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())

        # 定义优化器
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LR)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LR)

    @staticmethod
    def _build_actor(state_shape):
        fruits_input = layers.Input(shape=state_shape[0])
        hidden = layers.Conv2D(32, (8, 8), (2, 2), activation='relu')(fruits_input)
        # hidden = layers.MaxPooling2D((3, 3))(hidden)
        hidden = layers.Conv2D(64, (5, 5), (2, 2), activation='relu')(hidden)
        # hidden = layers.MaxPooling2D((3, 3))(hidden)
        hidden = layers.Conv2D(64, (3, 3), (2, 2), activation='relu')(hidden)
        hidden = layers.Flatten()(hidden)
        fruits_output = layers.Dense(128, activation='relu')(hidden)

        next_fruit_input = layers.Input(state_shape[1])
        hidden = layers.Dense(32, activation='relu')(next_fruit_input)
        next_fruit_output = layers.Dense(32, activation='relu')(hidden)

        concat = layers.Concatenate()([fruits_output, next_fruit_output])
        hidden = layers.Dense(64, activation='relu')(concat)
        action_output = layers.Dense(1, activation='tanh')(hidden)

        model = tf.keras.Model([fruits_input, next_fruit_input], action_output)
        # print(model.summary())
        return model

    @staticmethod
    def _build_critic(state_shape):
        fruits_input = layers.Input(shape=state_shape[0])
        hidden = layers.Conv2D(32, (8, 8), (2, 2), activation='relu')(fruits_input)
        # hidden = layers.MaxPooling2D((3, 3))(hidden)
        hidden = layers.Conv2D(64, (5, 5), (2, 2), activation='relu')(hidden)
        # hidden = layers.MaxPooling2D((3, 3))(hidden)
        hidden = layers.Conv2D(64, (3, 3), (2, 2), activation='relu')(hidden)
        hidden = layers.Flatten()(hidden)
        fruits_output = layers.Dense(128, activation='relu')(hidden)

        next_fruit_input = layers.Input(state_shape[1])
        hidden = layers.Dense(32, activation='relu')(next_fruit_input)
        next_fruit_output = layers.Dense(32, activation='relu')(hidden)

        action_input = layers.Input(shape=(1,))
        action_output = layers.Dense(32, activation='relu')(action_input)

        concat = layers.Concatenate()([fruits_output, next_fruit_output, action_output])
        hidden = layers.Dense(64, activation='relu')(concat)
        q_output = layers.Dense(1)(hidden)

        model = tf.keras.Model([fruits_input, next_fruit_input, action_input], q_output)
        # print(model.summary())
        return model

    def policy(self, state, noise):
        """进行一次决策，输入state输出action"""
        state_tensor = (tf.expand_dims(state[0], 0), tf.expand_dims(state[1], 0))
        action_output = (self.actor(state_tensor)).numpy()[0, 0]

        # 加入噪音后，若出界则重新加入噪音尝试，最多重试10次
        for _ in range(10):
            action = action_output + noise.get()
            if -1 < action < 1:
                break
        return action

    @tf.function
    def _update_target(self):
        """更新目标网络"""
        print('update_target: Tracing! 此提示应仅显示1次。')
        for (w_t, w) in zip(self.target_actor.variables, self.actor.variables):
            w_t.assign(w * TAU + w_t * (1 - TAU))
        for (w_t, w) in zip(self.target_critic.variables, self.critic.variables):
            w_t.assign(w * TAU + w_t * (1 - TAU))

    @tf.function
    def _update_main(self, state, action, reward, not_done, next_state):
        """更新现实网络"""
        print('update_main: Tracing! 此提示应仅显示1次。不知道为什么会显示两次')
        with tf.GradientTape() as tape:
            target_action = self.target_actor(next_state, training=True)
            target_value = reward + not_done * GAMMA * self.target_critic(
                [next_state, target_action], training=True
            )
            value = self.critic([state, action], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(target_value - value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actor_action = self.actor(state, training=True)
            actor_value = self.critic([state, actor_action], training=True)
            actor_loss = -tf.math.reduce_mean(actor_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        return critic_loss, actor_loss

    def learn(self, state, action, reward, not_done, next_state):
        state = (tf.convert_to_tensor(state[0]), tf.convert_to_tensor(state[1]))
        action = tf.convert_to_tensor(action)
        reward = tf.convert_to_tensor(reward)
        not_done = tf.convert_to_tensor(not_done)
        next_state = (tf.convert_to_tensor(next_state[0]), tf.convert_to_tensor(next_state[1]))

        results = self._update_main(state, action, reward, not_done, next_state)
        self._update_target()
        return [float(x) for x in results]

    def save_weights(self, path):
        print('Saving weights...', end='')
        self.actor.save_weights(os.path.join(path, 'actor'))
        self.critic.save_weights(os.path.join(path, 'critic'))
        self.target_actor.save_weights(os.path.join(path, 'target_actor'))
        self.target_critic.save_weights(os.path.join(path, 'target_critic'))
        print('Saved')


class Memory:
    """用于记录经验，进行经验回放"""

    def __init__(self, state_shape, buffer_size=10000, batch_size=64):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.state_buffer1 = np.zeros((self.buffer_size,) + state_shape[0], dtype='float32')
        self.state_buffer2 = np.zeros((self.buffer_size,) + state_shape[1], dtype='float32')
        self.action_buffer = np.zeros((self.buffer_size,), dtype='float32')
        self.reward_buffer = np.zeros((self.buffer_size,), dtype='float32')
        self.not_done_buffer = np.zeros((self.buffer_size,), dtype='float32')
        self.next_state_buffer1 = np.zeros((self.buffer_size,) + state_shape[0], dtype='float32')
        self.next_state_buffer2 = np.zeros((self.buffer_size,) + state_shape[1], dtype='float32')

        self.record_counter = 0

    def record(self, state, action, reward, not_done, next_state):
        """记录一条经验"""
        index = self.record_counter % self.buffer_size
        self.record_counter += 1

        self.state_buffer1[index], self.state_buffer2[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.not_done_buffer[index] = not_done
        self.next_state_buffer1[index], self.next_state_buffer2[index] = next_state

    def batch(self):
        """获取一组经验"""
        record_num = min(self.record_counter, self.buffer_size)
        batch_indices = np.random.choice(record_num, self.batch_size)

        state_batch = (self.state_buffer1[batch_indices], self.state_buffer2[batch_indices])
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        not_done_batch = self.not_done_buffer[batch_indices]
        next_state_batch = (self.next_state_buffer1[batch_indices], self.next_state_buffer2[batch_indices])

        return state_batch, action_batch, reward_batch, not_done_batch, next_state_batch


class Noise:

    def __init__(self):
        self.episode = 0

    def set_episode(self, episode):
        self.episode = episode

    def get(self):
        if self.episode > 300 or np.random.random() < self.episode / 150 - 1:  # 从第150个episode到第300个episode，返回0的概率从0增加到1
            return 0
        return np.random.randn() * 0.5 * (400 - self.episode) / 400


def train(url, browser, save_path):

    env = Environment(url, browser)
    agent = DDPGAgent(env.get_state_shape())
    memory = Memory(env.get_state_shape())
    noise = Noise()

    episode_results = []  # 每个episode的一些结果数据
    episode_written = 0  # 已保存结果数据数量

    for episode in range(EPISODE_NUM):
        state = env.reset()
        noise.set_episode(episode)
        results = []

        while True:
            action = agent.policy(state, noise)
            next_state, reward, done = env.step(action)

            memory.record(state, action, reward, not done, next_state)
            if episode < 10:  # 前10个episode不训练
                result = [0., 0.]
            else:
                result = agent.learn(*memory.batch())
            results.append(result)

            if done:
                break
            state = next_state

        episode_result = np.mean(results, axis=0)
        episode_result = np.append(env.score, episode_result)
        episode_results.append(episode_result)
        print('Episode {}: {}'.format(episode, episode_results[-1]))

        #  每隔10个episode保存一次模型权重和结果
        if (episode + 1) % 10 == 0 or episode + 1 == EPISODE_NUM:
            agent.save_weights(save_path)
            with open(RESULT_FILE, mode='a') as f:
                f.write(''.join(
                    ', '.join(str(x) for x in avg_result) + '\n'
                    for avg_result in episode_results[episode_written:]))
                episode_written = episode + 1


if __name__ == '__main__':
    import sys
    train(sys.argv[1], 'Chrome', 'model/test/')
