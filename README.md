# sparse-Q-learning
A multi-agent system Q-learning with sparse cooperation

Right now, the best joint action is computed with a naive method and not with the elimination algorithm : we just compute the payoff for every combination of the action and we take the best


### Launch

1. learn mode

Let the agents learn a policy during n episodes

```bash
python3 main.py learn [directory] -e episode -g grid -v
``` 

   * directory : directory to store the rules file
   * episode : number of episode
   * grid : grid size of the prey-predators game

2. play mode

Play the game with a learned policy

```bash
python3 main.py play [directory] -g grid
```

   * directory : directory to store the rules file
   * grid : grid size of the prey-predators game
  
3. test mode

Test the performance of the learning

```bash
python3 main.py test [directory] -e episode -r run -g grid -v
```
   * episode : number of episode
   * run : number of run
   * grid : grid size of the prey-predators game
  
### Results

<p align="center">
  <img src="https://github.com/amaurylekens/sparse-Q-learning/blob/master/images/result_4_4.png" style="width: 10%; height: 10%"/>
</p>

### Game in the console
<p align="center">
  <img src="https://github.com/amaurylekens/sparse-Q-learning/blob/master/images/game_console.png" style="width: 10%; height: 10%"/>
</p>

### Class diagram
<p align="center">
  <img src="https://github.com/amaurylekens/sparse-Q-learning/blob/master/images/class_diagram.png" style="width: 10%; height: 10%"/>
</p>
