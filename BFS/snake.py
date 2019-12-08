

# SNAKES GAME
# Use ARROW KEYS to play, SPACE BAR for pausing/resuming and Esc Key for exiting

import curses
import time
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint

curses.initscr()
win = curses.newwin(20, 60, 0, 0)
win.keypad(1)
curses.noecho()
curses.curs_set(0)
win.border(0)
win.nodelay(1)


def main():

    key = KEY_RIGHT  # Initializing snake start direction
    score = 0
    snake = [[10, 5], [10, 4], [10, 3]]  # Initial snake coordinates
    food = [randint(1, 18), randint(1, 58)]  # First food coordinates
    win.addch(food[0], food[1], '*')  # Prints the food

    current_time = 0

    while key != 27:  # While Esc key is not pressed
        path = None
        win.border(0)
        win.addstr(0, 2, ' Score: ' + str(score) + ' ')
        win.addstr(0, 27, ' Snake ')

        win.timeout(20)
        prev_key = key  # Previous direction choice
        event = win.getch()

        # FailSafe loop break.
        if event == 27:
            break

        # Pause & Resume. If SPACE BAR is pressed, wait for another press.
        if key == ord(' '):
            key = -1
            while key != ord(' '):
                key = win.getch()
            key = prev_key
            continue

        # If an invalid key is pressed
        if key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27]:
            key = prev_key

        path = search(create_grid(snake), (snake[0][0], snake[0][1]), (food[0], food[1]))
        # print("path", path)
        if path:
            if snake[0][0] < path[1][0]:  # SNAKE GOING DOWN
                event = KEY_DOWN

            if snake[0][0] > path[1][0]:  # SNAKE GOING UP
                event = KEY_UP

            if snake[0][1] > path[1][1]:  # SNAKE GOING LEFT
                event = KEY_LEFT

            if snake[0][1] < path[1][1]:  # SNAKE GOING RIGHT
                event = KEY_RIGHT

        key = key if event == -1 else event

        # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
        snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1),
                         snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])

        # Exit if snake crosses the boundaries
        if snake[0][0] == 0 or snake[0][0] == 19 or snake[0][1] == 0 or snake[0][1] == 59:
            break

        # Exit if snake runs over itself
        if snake[0] in snake[1:]:
            break

        # Respawn food if not consumed within 10 secs.
        if current_time != 0:
            if ((time.time() - current_time) % 60) > 10:  # When snake doesn't eat food within 10 seconds.
                win.addch(food[0], food[1], ' ')
                current_time = time.time()
                food = []
                while not food:
                    food = [randint(1, 18), randint(1, 58)]
                    if food in snake:
                        food = []
                win.addch(food[0], food[1], '*')

        # When snake consumes food. Else: Deletes end character of snake to create moving effect.
        if snake[0] == food:
            current_time = time.time()
            food = []
            score += 1
            while not food:
                food = [randint(1, 18), randint(1, 58)]
                if food in snake:
                    food = []
            win.addch(food[0], food[1], '*')
        else:
            last = snake.pop()
            win.addch(last[0], last[1], ' ')

        win.addch(snake[0][0], snake[0][1], 'o')

    curses.endwin()
    print("\nScore: " + str(score))


class Node:

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    # Returns Node's coordinate position with grid
    def __getitem__(self):
        return self.position


# Returns a list of tuples as a path from the given start to end positions within the given grid
def search(grid, start, end):

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        for new_position in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # This prevents hitting the walls
            if node_position[0] > (len(grid) - 1) \
                    or node_position[0] < 0 \
                    or node_position[1] > (len(grid[len(grid) - 1]) - 1) \
                    or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if grid[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1])
                                                                           ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            if child not in open_list and child not in closed_list:
                open_list.append(child)


def create_grid(snaking):

    grid = []

    # Creates the grid
    for x in range(20):
        grid.append([0] * 60)

    # Sets the snake position within the grid
    for z in range(len(snaking)):
        grid[snaking[z][0]][snaking[z][1]] = 1

    # Sets X coordinate boundaries
    for y in range(20):
        grid[y][0] = 1
        grid[y][59] = 1

    # Sets Y coordinate boundaries
    for h in range(59):
        grid[0][h] = 1
        grid[19][h] = 1

    return grid


if __name__ == '__main__':
    main()

