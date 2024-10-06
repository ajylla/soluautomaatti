import numpy as np
import matplotlib.pyplot as plt
import copy
from numpy.random import default_rng
import matplotlib.animation as ani
from time import time

class GameBoard:
    '''
    GameBoard class.
    Handles the creation, updating and animating of the game area.
    '''

    def __init__(self, size=[25, 25], board=None):

        # This list holds each iteration of the game
        # board to be animated later.
        self.history = []

        # If a board configuration is not given,
        # a random board is generated.
        if(board is not None):

            self.board = board

        else:

            self.board = self.make_random_board(size)

        # Saves a copy of the initial configuration of
        # the board.
        self.initial_board = copy.copy(self.board)

    def shift_positions(self, x, y):
        '''
        Shifts given x and y values such that they are inside the game area.
        Used for periodic boundary conditions.

        Args:
            x (int): x coordinate of a point
            y (int): y coordinate of a point

        Returns:
            int, int: shifted x and y coordinates.
        '''

        # Gets the dimensions of the board.
        x_len = self.get_size()[0]
        y_len = self.get_size()[1]

        # Shifting the values.
        while(x < 0):
            x += x_len

        while(x >= x_len):
            x -= x_len

        while(y < 0):
            y += y_len

        while(y >= y_len):
            y -= y_len

        return x, y

    def make_random_board(self, size):
        '''
        Generates a random board configuration.

        Args:
            size: dimensions of the board

        Returns:
            np.array: randomized board configuration
        '''

        # Initializing an array of zeros that matches
        # the dimensions.
        board = np.zeros((size[0], size[1]))

        # Rng initialization.
        rng = default_rng()

        # This variables determines the chance that
        # a cell is alive on the board.
        spawn_chance = 0.2

        # Loops over each cell.
        for row in range(len(board)):
            for col in range(len(board[0])):

                # Generates a random number and makes cells
                # alive if it is less than the spawn chance.
                rand = rng.random()
                if(rand <= spawn_chance):
                    board[row, col] = 1

        return board

    def get_board(self):
        '''
        Returns:
            np.array: current board configuration
        '''

        return self.board

    def get_size(self):
        '''
        Returns:
           list: dimensions of the board in format [x, y]
        '''

        return [len(self.board[0]), len(self.board)]

    def get_initial_board(self):
        '''
        Returns:
            np.array: initial configuration of the board
        '''

        return self.initial_board

    def get_neighbors(self, x, y):
        '''
        Gets the number of alive neighbors for a give x, y
        coordinate on the board.
        Checks the Moore neighborhood, that is, the 8 cells
        surrounding a cell.

        Args:
            x (int): x coordinate of cell
            y (int): y coordinate of cell

        Returns:
            int: number of alive neighbors (x, y) cell has
        '''

        n_neighbors = 0

        # Loops over the Moore neighborhood of the cell.
        # Also checks the cell itself but this is handled
        # later.
        for i in [x-1, x, x+1]:
            for j in [y-1, y, y+1]:

                # Shifts the positions such that periodic
                # boundary conditions apply.
                new_i, new_j = self.shift_positions(i, j)

                # Adds one the neighbor count if alive
                # cell is found.
                if(self.board[new_j, new_i] == 1):

                    n_neighbors += 1

        # Subtracts one from neighbor count if the cell
        # itself is alive.
        if(self.board[y, x] == 1):
            n_neighbors -= 1

        return n_neighbors

    def update_cell(self, x, y):
        '''
        Updates a single cell's status. Does not
        do it in-place but returns the result.
        The updating rules are as follows:

            1. If a alive cell has less than 2 or
               more than 3 alive neighbors, it will die.

            2. If a dead cell has exactly 3 alive neighbors,
               it will become alive.

            3. In all other cases, the status of a cell
               remains unchanged.

        Args:
            x (int): x coordinate of cell
            y (int): y coordinate of cell

        Returns:
            int: updates cell status i.e. 0 or 1.
        '''

        # Gets the number of alive neighbors the cell has.
        n_neighbors = self.get_neighbors(x, y)

        # Checks the status and neighbor conditions and
        # returns the appropriate status, according
        # to the rules above.
        if(self.board[y, x] == 0):

            if(n_neighbors == 3):

                return 1

        else:

            if(n_neighbors < 2 or n_neighbors > 3):

                return 0

        return self.board[y, x]

    def tick(self):
        '''
        Function to advance the game board by one timestep.
        One timestep consists of checking the rules for each
        cell first, and then updating them all at once.
        In practice, this is done with a temp board.
        Advancing the board is done in-place.
        '''

        # Appends a copy of the board configuration to the
        # history list.
        self.history.append(copy.copy(self.board))

        # Creates a temp board for which chances are
        # first made.
        temp_board = copy.copy(self.board)

        # Loops over the cells.
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):

                # Gets the updated status of each cell
                # and makes change to temp board.
                temp_board[y, x] = self.update_cell(x, y)

        # Applies all the changes made to the actual game board.
        self.board = temp_board

        # Finally, deletes temp_board to free up some memory.
        # Probably a pretty insignificant step.
        del temp_board

    def get_animation(self):
        '''
        Returns:
            list: list of all iterations of the board
        '''

        return self.history

    def draw(self, frame):
        '''
        Draw function that is used by animation function.

        Args:
            frame (int): index of frame in history list to be drawn
        '''

        # Clears the plot.
        plt.clf()

        # Initialized the axis and sets aspect ratio
        # such that x and y have the same scaling.
        ax = plt.axes()
        ax.set_aspect('equal')
        ax.set(xticks=[], yticks=[])

        # Inverts the y axis so the board appears the same
        # orientation as the board textfiles.
        plt.gca().invert_yaxis()

        # Plots a colormesh. Alive cells appear as black and dead ones white.
        plt.pcolormesh(self.history[frame], cmap='Greys', vmin=0, vmax=1)

    def animate(self, start_delay=None):
        '''
        Animates the board configuration history using draw function.
        Also saves the animation to a .gif file.
        '''

        if start_delay is not None:
            for _ in range(start_delay):
                self.history = [self.history[0]] + self.history

        # Number of frames is the length of history list.
        n_frames = len(self.history)

        # This variable essentially controls the speed of
        # the animation. It is the waiting interval
        # between frames, so a higher value means a slower
        # animation.
        interval = 400

        # Creates the figure on which to draw.
        fig = plt.figure()

        # Calls the matplotlib FuncAnimation function.
        # Animation must be stored in a variable even if it is
        # never used for the animation to play correctly.
        # The variable is destroyed when animation window is
        # closed.
        animation = ani.FuncAnimation(fig, self.draw, n_frames,
                                      interval=interval)

        # Saves the animation as a .gif file. Requires
        # some video transcoder such as ffmpeg or Pillow.
        animation.save("animation.gif")

        plt.show()


def read_board(filename):
    '''
    Reads board from a file. A board file be a .txt file
    in a format of 0's and 1's separated by one whitespace
    or a line change for the next row of cells. 0's represent
    a dead cell and 1's represent a alive cell.
    Example of what a file should look like:

    0 1 0 0 0 0 1
    1 0 0 0 0 1 0
    0 0 0 0 0 0 0
    1 1 0 0 0 0 0

    Args:
        filename (str): name of file to read

    Returns:
        np.array: a board array
    '''

    # Opens file for reading.
    _file = open(filename, 'r')

    # Reads line by line and appends lines to list.
    lines = _file.readlines()

    # Initializing board array.
    board = []

    # Iterates through the lines list.
    for line in lines:
        # Splits a line at whitespace.
        split = line.split()

        # Converts each 0 and 1 to an integer
        # and creates a list.
        split = [int(x) for x in split]

        # Appends list of 0's and 1's
        # to board list.
        board.append(split)

    # Closes file and returns
    # np.array.
    _file.close()
    return np.array(board)


def save_board(board, filename):
    '''
    Saves board array to a file. Output file format
    matches that of input to read_board.

    Args:
        board (np.array): board array to save
        filename (str): name of output file
    '''

    # Opens file for writing/overwriting.
    _file = open(filename, 'w')

    # Initialize string to be written to file.
    board_txt = ""

    # Loops through the board.
    for y in range(len(board)):
        for x in range(len(board[0])):

            # Appends the cell status and a space
            # to the string.
            board_txt = board_txt + str(int(board[y, x])) + " "

        # Adds a newline when end of a row is reached.
        board_txt = board_txt + "\n"

    # Writes string to file and closes file.
    _file.write(board_txt)
    _file.close()


def print_progress(step, total):
    '''
    Prints a progress bar.

    Args:
        step (int): current step/progress of the program
        total (int): total steps the program will run for
    '''

    # Creating the string to print, total length of the bar
    # and number of spots to fill on bar variables.
    message = "Simulating... ["
    total_bar_length = 60
    percentage = int(step / total * 100)
    bar_fill = int(step / total * total_bar_length)

    # Iterating over the bar length, if spot needs to be filled,
    # appends | to the string, whitespace is appended for rest.
    for i in range(total_bar_length):
        if i < bar_fill:
            message += "|"
        else:
            message += " "

    # Appends closing bracked and the percentage value.
    message += "] " + str(percentage) + " %"

    # If bar is not full, end the line in a carriage
    # return so it prints on the same line every time.
    if step < total:
        print(message, end='\r')
    else:
        print(message)


def main(n_steps, size=[50, 50], filename=None, animate=True, start_delay=None):
    '''
    Main function of the program. Creates the GameBoard object, runs
    the simulation and saves the initial configuration of the board.

    Args:
        n_steps (int): number of timesteps to simulate
        size (list): size of the board in format 
                     [horizontal length, vertical length] (default: [50, 50])
        filename (str): name of a file to load (default: None)
        animate (bool): toggle animation (default: True)
        start_delay (int): start delay of the animation in frames

    Returns:
        float: simulation time in seconds
    '''

    # Start time of simulation.
    start_time = time()

    # Initialized the GameBoard object with or without
    # a board configuration, depending on input.
    if(filename is not None):
        board = read_board(filename)
        gameBoard = GameBoard(size, board)
    else:
        gameBoard = GameBoard(size)

    # Advances the board by given timesteps and
    # prints and updates the progress bar.
    for i in range(n_steps+1):
        gameBoard.tick()
        print_progress(i, n_steps+1)

    # Prints full progress bar.
    print_progress(n_steps, n_steps)

    # Gets and saves the initial configuration of the
    # board.
    # Note that the file is overwritten every time, unless
    # a new name is specified.
    initial_board = gameBoard.get_initial_board()
    save_board(initial_board, 'initial-board.txt')

    # End time of simulation. Prints time taken.
    # This does not take into account the time to
    # save the .gif file, which is surprisingly slow.
    end_time = time()
    delta_time = end_time - start_time
    print("Took: " + str(round(delta_time, 1)) + " s.")

    # Animates the board. This also saves the animation as
    # a .gif file.
    # Note that this file is also overwritten each time.
    if(animate):
        gameBoard.animate(start_delay=start_delay)

    return delta_time

# Main guard.
if(__name__ == "__main__"):

    #main(100, filename='glider.txt')
    #main(100, filename='glider-gun.txt')
    #main(20, filename='pulsar.txt')
    main(100, size=[50, 50], animate=True, start_delay=1)
