import pygame
from pygame.locals import *
import argparse

from agent import Agent
from snake import SnakeEnv
import utils
import time


class Application:
    def __init__(self):
        self.env = SnakeEnv(utils.snake_head_x, utils.snake_head_y, utils.food_x, utils.food_y)
        self.agent = Agent(self.env.get_actions(), utils.Ne, utils.C, utils.gamma)
        
    def execute(self):
        print("inside execute")
        if not utils.human:
            if utils.train_eps != 0:
                self.train()
            self.test()
        self.show_games()

    def train(self):
        print("Train Phase:")
        self.agent.train()
        window = utils.window
        self.points_results = []
        first_eat = True
        start = time.time()

        for game in range(1, utils.train_eps + 1):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            while not dead:
                state, points, dead = self.env.step(action)
                if first_eat and points == 1:
                    self.agent.save_model(utils.CHECKPOINT)
                    first_eat = False
                action = self.agent.act(state, points, dead)

            points = self.env.get_points()
            self.points_results.append(points)
            if game % utils.window == 0:
                print(
                    "Games:", len(self.points_results) - window, "-", len(self.points_results), 
                    "Points (Average:", sum(self.points_results[-window:])/window,
                    "Max:", max(self.points_results[-window:]),
                    "Min:", min(self.points_results[-window:]), ")",
                )
            self.env.reset()
        print("Training takes", time.time() - start, "seconds")
        self.agent.save_model(utils.model_name)

    def test(self):
        print("Test Phase:")
        self.agent.eval()
        self.agent.load_model(utils.model_name)
        points_results = []
        start = time.time()

        for game in range(1, utils.test_eps + 1):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            while not dead:
                state, points, dead = self.env.step(action)
                action = self.agent.act(state, points, dead)
            points = self.env.get_points()
            points_results.append(points)
            self.env.reset()

        print("Testing takes", time.time() - start, "seconds")
        print("Number of Games:", len(points_results))
        print("Average Points:", sum(points_results)/len(points_results))
        print("Max Points:", max(points_results))
        print("Min Points:", min(points_results))

    def show_games(self):
        print("Display Games")
        self.env.display()
        pygame.event.pump()
        self.agent.eval()
        points_results = []
        end = False
        for game in range(1, utils.show_eps + 1):
            state = self.env.get_state()
            dead = False
            action = self.agent.act(state, 0, dead)
            count = 0
            while not dead:
                count += 1
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[K_ESCAPE] or self.check_quit():
                    end = True
                    break
                state, points, dead = self.env.step(action)
                # Qlearning agent
                if not utils.human:
                    action = self.agent.act(state, points, dead)
            if end:
                break
            self.env.reset()
            points_results.append(points)
            print("Game:", str(game)+"/"+str(utils.show_eps), "Points:", points)
        if len(points_results) == 0:
            return
        print("Average Points:", sum(points_results)/len(points_results))

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False


def main():
    app = Application()
    app.execute()


if __name__ == "__main__":
    main()
