
from direct_keys import PressKey, ReleaseKey, ESC, game_keys
from process_image import *
from collections import deque


def pause_button():
    PressKey(ESC)
    ReleaseKey(ESC)


def press_game_key(key):
    if key == 0:
        PressKey(game_keys[0])
    elif key == 1:
        PressKey(game_keys[1])
    elif key == 2:
        PressKey(game_keys[2])
    elif key == 3:
        PressKey(game_keys[3])
    elif key == 4:
        PressKey(game_keys[0])
        PressKey(game_keys[2])
    elif key == 5:
        PressKey(game_keys[1])
        PressKey(game_keys[2])
    elif key == 6:
        PressKey(game_keys[0])
        PressKey(game_keys[2])
        PressKey(game_keys[3])
    elif key == 7:
        PressKey(game_keys[1])
        PressKey(game_keys[2])
        PressKey(game_keys[3])
    elif key == 8:
        PressKey(game_keys[0])
        PressKey(game_keys[3])
    elif key == 9:
        PressKey(game_keys[1])
        PressKey(game_keys[3])


def release_every_key():
    for key in game_keys:
        ReleaseKey(key)


def get_state():
    state = process_img(get_image())
    return state


def get_state_arr(state_deque, start, stop):
    state_array = np.array(list(state_deque)[start:stop])
    state_array = np.transpose(state_array)
    return state_array


def get_reward(state_deque):
    if np.sum(state_deque[-1][8:15, 14:34] - state_deque[-2][8:15, 14:34]) != 0:  # detect score change
        return 1
    else:
        return 0


def end_of_episode(state):
    if state[1, 1] == 0:  # detect screen color(pure black) when mario dies
        return 1
    else:
        return 0


def start_of_episode(state):
    if state[1, 1] != 0:  # detect screen color when game starts
        return 1
    else:
        return 0


if __name__ == '__main__':
    state_deque = deque(maxlen=5)

    release_every_key()

    while True:
        state = get_state()
        state_deque.append(state)

        if end_of_episode(state):
            continue

        if len(state_deque) == 5 and get_reward(state_deque) == 1:
            print('+Reward')

        cv2.imshow('mario', state)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
