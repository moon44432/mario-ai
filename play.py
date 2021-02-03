import random
import time
from collections import deque

import numpy as np
from keras.models import load_model

from env import get_state_arr, get_state, release_every_key, \
    start_of_episode, end_of_episode, do_action, action_name, action_size
from hparams import num_gameplay, skip_frames, state_deque_size, exp_rate
from process_image import img_width

if __name__ == '__main__':
    model = load_model('./model/model.h5')
    model.summary()

    for game in range(1, num_gameplay + 1):
        while True:
            state = get_state()
            if start_of_episode(state) == 1:
                break

        step, action, value, epsilon = 0, 1, 0, exp_rate
        state_deque = deque(maxlen=state_deque_size)

        while True:
            step += 1
            current_state_arr = get_state_arr(state_deque)

            if step % skip_frames == 1:
                release_every_key()

                if epsilon > np.random.rand() or len(state_deque) < state_deque_size:
                    action = random.randrange(0, action_size)
                else:
                    predict_arr = np.zeros((1, img_width, img_width, state_deque_size))
                    predict_arr[0] = current_state_arr
                    predict = model.predict(predict_arr)[0]
                    action = np.argmax(predict)
                    value = predict[action]
                    print(predict)

                do_action(action)
                time.sleep(0.05)

            state = get_state()
            state_deque.append(state.astype('float32') / 255.0)

            print('Game #{} Step: {} Action: {} Value: {}'.format(game, step, action_name[action], value))

            time.sleep(0.05)

            if end_of_episode(state):
                break

    del model
