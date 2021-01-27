
import time
import random
from network import set_network, action_size
from env import *
from collections import deque
from tensorflow.keras.models import load_model

# params

NUM_GAMES = 1000
MAX_STEPS = 20000
GAMMA = 0.99
WARMUP = 10
SKIP_FRAMES = 4


if __name__ == '__main__':
    model = load_model('./model/model.h5')

    for game in range(1, NUM_GAMES + 1):
        while True:
            state = get_state()
            if start_of_episode(state) == 1:
                break

        step = 0
        action = 1
        epsilon = 0.1
        state_deque = deque(maxlen=5)

        for _ in range(1, MAX_STEPS + 1):
            step += 1
            current_state_arr = get_state_arr(state_deque, 0, 4)

            if step % SKIP_FRAMES == 1:
                release_every_key()

                if epsilon > np.random.rand() or len(state_deque) < 5:
                    action = random.randrange(0, action_size)
                else:
                    predict_arr = np.zeros((1, 128, 128, 4))
                    predict_arr[0] = current_state_arr
                    action = np.argmax(model.predict(predict_arr)[0])

                press_game_key(action)

            state = get_state()
            state_deque.append(state)

            print('Game #{} Step: {} Action: {} '.format(game, step, action), end='')
            print('')

            time.sleep(0.05)

            if end_of_episode(state):
                break