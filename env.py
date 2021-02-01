
from direct_keys import PressKey, ReleaseKey, ESC, game_keys
from process_image import *
from collections import deque

action_name = {0: 'Nothing', 1: 'Left', 2: 'Right', 3: 'Right + Jump', 4: 'Jump'}


def pause_button():
    PressKey(ESC)
    ReleaseKey(ESC)


def do_action(key):
    if key == 0:
        pass  # Nothing
    elif key == 1:
        PressKey(game_keys[0])  # Left
    elif key == 2:
        PressKey(game_keys[1])  # Right
    elif key == 3:
        PressKey(game_keys[1])  # Right + Jump
        PressKey(game_keys[3])
    elif key == 4:
        PressKey(game_keys[3])  # Jump


def release_every_key():
    for key in game_keys:
        ReleaseKey(key)


def get_state():
    state = process_img(get_image())
    return state


def get_state_arr(state_deque):
    state_array = np.array(list(state_deque))
    state_array = np.transpose(state_array)
    return state_array


def is_scrolling(state_deque):
    if np.sum(state_deque[-1][120:128, 0:128] - state_deque[-2][120:128, 0:128]) != 0:  # detect screen scrolling
        return 1
    else:
        return 0


def get_reward(state_deque):
    if np.sum(state_deque[-1][8:15, 14:34] - state_deque[-2][8:15, 14:34]) != 0:  # detect score change
        return 1
    else:
        return 0


def is_dead(state_deque):
    if np.sum(state_deque[-1][0:10, 120:128] - state_deque[-2][0:10, 120:128]) != 0:  # detect death message
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
    state_deque = deque(maxlen=4)
    release_every_key()

    while True:
        state = get_state()
        state_deque.append(state)

        if end_of_episode(state):
            continue

        if len(state_deque) == 4:
            if get_reward(state_deque) == 1:
                print('+Reward')
            if is_scrolling(state_deque):
                # print('Scrolling')
                pass
            if is_dead(state_deque):
                print('Dead')

        cv2.imshow('mario', state)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
