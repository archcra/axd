

# SELF PLAY
EPISODES = 100
MCTS_SIMS = 50
MEMORY_SIZE = 6000
TURNS_UNTIL_TAU0 = 10  # turn on which it starts playing deterministically
CPUCT = 2
EPSILON = 0.2
ALPHA = 0.8


# RETRAINING
BATCH_SIZE = 512
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
    {'filters': 256, 'kernel_size': (5, 5)}, {'filters': 256, 'kernel_size': (3, 3)}, {'filters': 256, 'kernel_size': (3, 3)},
    {'filters': 256, 'kernel_size': (3, 3)},{'filters': 256, 'kernel_size': (3, 3)},{'filters': 256, 'kernel_size': (3, 3)},
    {'filters': 256, 'kernel_size': (3, 3)},{'filters': 256, 'kernel_size': (3, 3)}
]

# EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3
