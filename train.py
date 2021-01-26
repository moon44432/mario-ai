
import numpy as np
from network import set_network
from exp_memory import Memory
from collections import deque
from tensorflow.losses import huber_loss
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

E_START = 1.0
E_STOP = 0.01
E_DECAY_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 32


if __name__ == '__main__':
    set_network()

    main_qn = load_model('./model/model.h5')
    target_qn = load_model('./model/model.h5')
    main_qn.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
    target_qn.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')
    memory = Memory(MEMORY_SIZE)

    total_step = 0

    for episode in range(1, NUM_EPISODES + 1):
        step = 0

        # set weights of target Q-Network
        target_qn.model.set_weights(main_qn.model.get_weights())

        for _ in range(1, MAX_STEPS + 1):
            step += 1
            total_step += 1

            # epsilon decay
            epsilon = E_STOP + (E_START - E_STOP) * np.exp(-E_DECAY_RATE * total_step)

            if epsilon > np.random.rand():
                action = env.action_space.sample()
            else:
                action = np.argmax(main_qn.model.predict(state)[0])

            next_state, _, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            if done:
                if step >= 190:
                    success_count += 1
                    reward = 1
                else:
                    success_count = 0
                    reward = 0

                # empty state
                next_state = np.zeros(state.shape)

                if step > WARMUP:
                    memory.add((state, action, reward, next_state))
            else:
                reward = 0

                if step > WARMUP:
                    memory.add((state, action, reward, next_state))

                state = next_state

            if len(memory) >= BATCH_SIZE:
                inputs = np.zeros((BATCH_SIZE, 4))  # input (state)
                targets = np.zeros((BATCH_SIZE, 2))  # output (value of each action)

                minibatch = memory.sample(BATCH_SIZE)

                for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
                    inputs[i] = state_b

                    # compute value
                    if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                        target = reward_b + GAMMA * np.amax(target_qn.model.predict(next_state_b)[0])
                    else:
                        target = reward_b

                    targets[i] = main_qn.model.predict(state_b)
                    targets[i][action_b] = target

                main_qn.model.fit(inputs, targets, nb_epoch=1, verbose=0)

            if done:
                break

        print('에피소드: {}, 스텝 수: {}, epsilon: {:.4f}'.format(episode, step, epsilon))

        # stop learning when successes 5 times in a row
        if success_count >= 5:
            break

        state = env.reset()
        state = np.reshape(state, [1, state_size])