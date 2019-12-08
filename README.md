# AI Snake BFS & QLearning 🐍

Written in Python 3.6

Base of game is taken from [Sanchit Gangwar](https://gist.github.com/sanchitgangwar/2158089)

## Game Play
![](snake.gif)

## Requirements: 
Pygame, Numpy


## BFS
- bfs_snake.py      ---  BFS Algorithm for snake

To Run: In a console, go to the project directory and issue: 'python snake.py'

## RL
- main.py           ---  main program
- snake.py          ---  normal snake pyGame
- agent.py          ---  Q-learning agent

To Run: In a console, go to the project directory and issue: 'python main.py'

## Experiments

We used two algorithms to evaluate the performance of an AI:

1. **BFS**
2. **RL Q Learning**

Test results (averaged over 10000 episodes):

| Solver | Average Length | Max Length |
| :----: | :------------: | :-----------: |
|BFS|81.93|89|
|Q Learning|27.50|58|
