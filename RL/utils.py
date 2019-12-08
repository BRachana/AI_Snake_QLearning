import numpy as np

DISPLAY_SIZE = 560
GRID_SIZE = 40
WALL_SIZE = 40

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (72, 61, 139)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)


train_eps = 10000   # Number of Training Episodes
test_eps = 1000
show_eps = 10
window = 100
human = False
gameOver = 0


C = 85              # learning rate
Ne = 10             # randomize
gamma = 0.7         # discount Factor
snake_head_x, snake_head_y, food_x, food_y = 80, 80, 200, 200


CHECKPOINT = 'checkpoint.npy'
model_name = 'q_agent.npy'

NUM_ADJOINING_WALL_X_STATES = 3
NUM_ADJOINING_WALL_Y_STATES = 3
NUM_FOOD_DIR_X = 3
NUM_FOOD_DIR_Y = 3
NUM_ADJOINING_BODY_TOP_STATES = 2
NUM_ADJOINING_BODY_BOTTOM_STATES = 2
NUM_ADJOINING_BODY_LEFT_STATES = 2
NUM_ADJOINING_BODY_RIGHT_STATES = 2
NUM_ACTIONS = 4


def create_q_table():
    return np.zeros((NUM_ADJOINING_WALL_X_STATES, NUM_ADJOINING_WALL_Y_STATES,
                     NUM_FOOD_DIR_X, NUM_FOOD_DIR_Y,
                     NUM_ADJOINING_BODY_TOP_STATES, NUM_ADJOINING_BODY_BOTTOM_STATES,
                     NUM_ADJOINING_BODY_LEFT_STATES, NUM_ADJOINING_BODY_RIGHT_STATES,
                     NUM_ACTIONS))


def sanity_check(arr):
    if (type(arr) is np.ndarray and
            arr.shape == (NUM_ADJOINING_WALL_X_STATES, NUM_ADJOINING_WALL_Y_STATES,
                          NUM_FOOD_DIR_X, NUM_FOOD_DIR_Y,
                          NUM_ADJOINING_BODY_TOP_STATES, NUM_ADJOINING_BODY_BOTTOM_STATES,
                          NUM_ADJOINING_BODY_LEFT_STATES, NUM_ADJOINING_BODY_RIGHT_STATES,
                          NUM_ACTIONS)):
        return True
    else:
        return False


def save(filename, arr):
    if sanity_check(arr):
        np.save(filename, arr)
        return True
    else:
        print("Failed to save model")
        return False


def load(filename):
    try:
        arr = np.load(filename)
        if sanity_check(arr):
            print("Loaded model successfully")
            return arr
        print("Model loaded is not in the required format")
        return None
    except:
        print("Filename doesnt exist")
        return None
