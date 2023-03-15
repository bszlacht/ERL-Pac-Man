# Combining different learned agents to solve Pac-Man

## General idea:
We want to check if there are any benefits of using ensemble learning and compare it to methods that are used to build
the ensemble learning model. <br />
![alt_text](https://github.com/bszlacht/ERL-Pac-Man/blob/main/readme_imgs/1.jpg)<br />

## What is ensemble learning?

The idea behind ensemble learning is that by combining the predictions of multiple models, 
the errors of individual models can be offset, leading to more accurate and reliable predictions.
In the context of reinforcement learning, ensemble learning can be used to improve the performance of a reinforcement 
learning agent.

One way to use ensemble learning in reinforcement learning is to train multiple agents with different initial 
conditions or hyperparameters. These agents can then be combined by taking the average or weighted sum of their 
Q-values or policy outputs. This can help reduce the impact of overfitting and bias in the individual agents, 
resulting in more stable and accurate performance.

Ensemble learning can also be used to improve the exploration-exploitation tradeoff in reinforcement learning. 
This involves using a combination of exploration strategies, such as epsilon-greedy, upper confidence bound, 
and Thompson sampling, to balance the tradeoff between trying new actions and exploiting the current policy.

## Ensemble learning in solving Pac-Man? 

Our idea is to train different models with different policies and different algorithms like Sarsa, Double Q-learning
and MCTS to then combine them into one ensemble learning model.

## Scope:

We want to implement few reinforcement learning models and use them as agents in order to combine them in a 
single model. The problem to solve will be a popular game Pac-Man. We chose it because it is simple to visualize 
and the rules of the game are clear.

## Plan:

1. Implement Pac-Man environment.
2. Pac-Man visualization in PyGame.
3. Implement few reinforcement learning models.
4. Train previously created models.
5. Implement ensemble learning model.
6. Use ensemble learning model.
7. Graph results and compare ensemble vs single rl models.

## Requirements:
