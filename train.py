
import random
import time
from network import set_network, action_size
from exp_memory import Memory
from env import *
from collections import deque
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# params

NUM_EPISODES = 1000
MAX_STEPS = 20000
GAMMA = 0.99
WARMUP = 10

E_START = 0.62
E_STOP = 0.1
E_DECAY_RATE = 0.00001

MEMORY_SIZE = 100000
BATCH_SIZE = 32

SKIP_FRAMES = 4


if __name__ == '__main__':
    set_network()

    main_qn = load_model('./model/model.h5')
    main_qn.compile(loss='categorical_crossentropy', optimizer='adam')

    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    memory = Memory(MEMORY_SIZE)

    total_step = 0

    for episode in range(1, NUM_EPISODES + 1):
        while True:
            state = get_state()
            if start_of_episode(state) == 1:
                break

        step = 0
        action = 1
        state_deque = deque(maxlen=5)
        dead = False
        do_learn = True

        for _ in range(1, MAX_STEPS + 1):
            step += 1
            total_step += 1

            # epsilon decay
            epsilon = E_STOP + (E_START - E_STOP) * np.exp(-E_DECAY_RATE * total_step)

            current_state_arr = get_state_arr(state_deque, 0, 4)

            if step % SKIP_FRAMES == 1:
                release_every_key()

                if epsilon > np.random.rand() or len(state_deque) < 5:
                    action = random.randrange(0, action_size)
                else:
                    predict_arr = np.zeros((1, 128, 128, 4))
                    predict_arr[0] = current_state_arr
                    action = np.argmax(main_qn.predict(predict_arr)[0])

                press_game_key(action)

            state = get_state()
            state_deque.append(state)

            print('Episode: {} Step: {} Action: {} '.format(episode, step, action), end='')
            if step > WARMUP:
                reward = get_reward(state_deque)
                if is_scrolling(state_deque):
                    reward += 0.1  # additional reward for moving toward the right side; especially for the Super Mario Bros
                if is_dead(state_deque):
                    reward -= 1
                    dead = True
                print('Reward: {} '.format(reward), end = '')
                memory.add((current_state_arr, action, reward, get_state_arr(state_deque, 1, 5)))

            if do_learn is True:
                if len(memory) >= BATCH_SIZE:
                    pause_button()

                    inputs = np.zeros((BATCH_SIZE, 128, 128, 4))  # input (state)
                    targets = np.zeros((BATCH_SIZE, action_size))  # output (value of each action)

                    minibatch = memory.sample(BATCH_SIZE)
                    print('Fitting...', end='')
                    for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                        inputs[i] = state_b

                        # compute value
                        if not (next_state_b[1][1][-1] == 0):
                            predict_arr = np.zeros((1, 128, 128, 4))
                            predict_arr[0] = next_state_b
                            target = reward_b + GAMMA * np.amax(main_qn.predict(predict_arr)[0])
                        else:
                            target = reward_b

                        predict_arr = np.zeros((1, 128, 128, 4))
                        predict_arr[0] = state_b

                        targets[i] = main_qn.predict(predict_arr)
                        targets[i][action_b] = target
                    main_qn.fit(inputs, targets, epochs=1, verbose=0)

                    pause_button()
                    time.sleep(0.05)

            print('')

            if dead is True:
                do_learn = False

            if end_of_episode(state):
                break

        print('에피소드: {}, 스텝 수: {}, epsilon: {:.5f}'.format(episode, step, epsilon))
        main_qn.save('./model/model.h5')

    K.clear_session()
    del main_qn
