import random
import time
from collections import deque

import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.losses import huber_loss

from env import get_state_arr, get_state, get_reward, release_every_key, \
    start_of_episode, end_of_episode, do_action, action_name, action_size, is_scrolling, is_dead, pause_button
from exp_memory import Memory
from hparams import memory_size, num_episodes, max_steps, epsilon_start, epsilon_stop, epsilon_decay_rate, \
    skip_frames, warmup_steps, batch_size, gamma, state_deque_size
from network import set_network, create_network
from process_image import img_width

if __name__ == '__main__':
    set_network()
    main_qn = load_model('./model/model.h5')
    main_qn.compile(loss=huber_loss, optimizer=Adam(lr=0.0005))
    target_qn = create_network()

    memory = Memory(memory_size)

    total_step = 0

    for episode in range(1, num_episodes + 1):
        while True:
            state = get_state()
            if start_of_episode(state) == 1:
                break

        step, action, value = 0, 1, 0
        do_learn, dead = True, False
        state_deque = deque(maxlen=state_deque_size)

        target_qn.model.set_weights(main_qn.model.get_weights())

        for _ in range(1, max_steps + 1):
            step += 1
            total_step += 1

            # epsilon decay
            epsilon = epsilon_stop + (epsilon_start - epsilon_stop) * np.exp(-epsilon_decay_rate * total_step)

            current_state_arr = get_state_arr(state_deque)

            if step % skip_frames == 1:
                release_every_key()

                if epsilon > np.random.rand() or len(state_deque) < state_deque_size:
                    action = random.randrange(0, action_size)
                    value = 0
                else:
                    predict_arr = np.zeros((1, img_width, img_width, state_deque_size))
                    predict_arr[0] = current_state_arr
                    predict = main_qn.predict(predict_arr)[0]
                    action = np.argmax(predict)
                    value = predict

                do_action(action)
                time.sleep(0.05)

            state = get_state()
            state_deque.append(state.astype('float32') / 255.0)

            print(
                'Episode: {}, Step: {}, Epsilon: {}, Action: {}, '.format(episode, step, epsilon, action_name[action]),
                end='')

            if step > warmup_steps:
                reward = get_reward(state_deque)
                if is_scrolling(state_deque) and (action == 2 or action == 3):
                    reward += 0.1  # additional reward for moving toward the right side
                if is_dead(state_deque):
                    reward = -1
                    dead = True
                print('Reward: {}'.format(reward))
                memory.add((current_state_arr, action, reward, get_state_arr(state_deque)))

            print(value)

            if do_learn is True:
                if len(memory) >= batch_size:
                    pause_button()

                    inputs = np.zeros((batch_size, img_width, img_width, state_deque_size))  # input (state)
                    targets = np.zeros((batch_size, action_size))  # output (value of each action)

                    minibatch = memory.sample(batch_size)
                    for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                        inputs[i] = state_b

                        # compute value
                        predict_arr = np.zeros((1, img_width, img_width, state_deque_size))
                        predict_arr[0] = next_state_b
                        target = reward_b + gamma * np.amax(target_qn.predict(predict_arr)[0])

                        predict_arr[0] = state_b
                        targets[i] = main_qn.predict(predict_arr)
                        targets[i][action_b] = target

                    main_qn.fit(inputs, targets, epochs=1, verbose=0)

                    time.sleep(0.1)
                    pause_button()
                    time.sleep(0.05)

            if dead is True:
                do_learn = False

            if end_of_episode(state):
                break

        main_qn.save('./model/model.h5')

    del main_qn
    del target_qn
