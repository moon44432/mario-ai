
import time
import random
from network import action_size
from env import *
from collections import deque
from keras.models import load_model


# params

NUM_GAMES = 1000
MAX_STEPS = 20000
GAMMA = 0.99
WARMUP = 10
SKIP_FRAMES = 4


if __name__ == '__main__':
    model = load_model('./model/model.h5')
    model.summary()

    for game in range(1, NUM_GAMES + 1):
        while True:
            state = get_state()
            if start_of_episode(state) == 1:
                break

        step = 0
        action = 1
        epsilon = 0.05
        value = 0
        state_deque = deque(maxlen=4)

        while True:
            step += 1
            current_state_arr = get_state_arr(state_deque)

            if step % SKIP_FRAMES == 1:
                release_every_key()

                if epsilon > np.random.rand() or len(state_deque) < 4:
                    action = random.randrange(0, action_size)
                else:
                    predict_arr = np.zeros((1, 128, 128, 4))
                    predict_arr[0] = current_state_arr
                    predict = model.predict(predict_arr)[0]
                    action = np.argmax(predict)
                    value = predict[action]
                    print(predict)

                do_action(action)
                time.sleep(0.05)

            state = get_state()
            cv2.imshow('mario', state)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

            state_deque.append(state.astype('float32') / 255.0)

            print('Game #{} Step: {} Action: {} Value: {}'.format(game, step, action, value), end='')
            print('')

            time.sleep(0.05)

            if end_of_episode(state):
                break
