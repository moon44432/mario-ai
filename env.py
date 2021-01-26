
from direct_keys import PressKey, ReleaseKey, ESC
from process_image import *
from collections import deque

def pause_button():
    PressKey(ESC)
    ReleaseKey(ESC)

def get_state():
    state = process_img(get_image())
    return state

def get_state_arr(state_deque):
    state_array = np.array(list(state_deque)[0:5])
    state_array = np.transpose(state_array)
    return state_array

def get_reward(state_deque):
    if len(state_deque) < 2:
        return 0
    if np.sum(state_deque[-1][8:15, 12:34] - state_deque[-2][8:15, 12:34]) != 0:
        return 1
    else:
        return 0

def end_of_episode(state):
    if state[0, 0] == 0:
        return 1
    else:
        return 0

if __name__ == '__main__':
    state_deque = deque(maxlen=5)

    while True:
        state = get_state()
        state_deque.append(state)

        if end_of_episode(state):
            continue

        if get_reward(state_deque) == 1:
            print('+Reward')

        # cv2.imshow('mario', state)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break