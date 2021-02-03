# image processing

mon = {'top': 60, 'left': 0, 'width': 512, 'height': 480}  # pos & size of emulator screen

# gameplay

num_gameplay = 1000
max_play_steps = 20000
skip_frames = 4
exp_rate = 0.1

# training

state_deque_size = 4

num_episodes = 1000
max_steps = 20000
gamma = 0.99
warmup_steps = 10

epsilon_start = 0.99
epsilon_stop = 0.1
epsilon_decay_rate = 0.000001

memory_size = 1000000
min_memory_len = 1000  # minimum replay memory length for training; must be larger than batch size
batch_size = 32
