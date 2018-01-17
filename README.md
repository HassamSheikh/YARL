
# YARL: Yet Another Reinforcement Learning Package
`YARL` is a Deep Reinforcement Learning package that is specifically designed to seamlessly integrate with [Keras](http://keras.io).
`YARL` works with [OpenAI Gym](https://gym.openai.com/) which means that it makes really easy to test same algorithm on multiple environments without having to change the code. `YARL` works with both Windows and Linux. As of now `YARL` supports only [TensorFlow](https://www.tensorflow.org/).

<table>
  <tr>
    <td><img src="/assets/cartpole.gif?raw=true" width="200"></td>
    <td><img src="/assets/pendulum.gif?raw=true" width="200"></td>
  </tr>
</table>


## Algorithms Implemented
The following algorithms have been implemented:

- Deep Q Learning (DQN) [[1]](http://arxiv.org/abs/1312.5602), [[2]](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)
- Double Deep Q Learning (DDQN) [[3]](http://arxiv.org/abs/1509.06461)
- Deep Deterministic Policy Gradient (DDPG) [[4]](http://arxiv.org/abs/1509.02971) (Work in progress)


## Requirements
- Python 3.5
- [Keras](http://keras.io) >= 1.0.7
- [OpenAI Gym](https://github.com/openai/gym)
- [h5py](https://pypi.python.org/pypi/h5py)


## References
1. *Playing Atari with Deep Reinforcement Learning*, Mnih et al., 2013
2. *Human-level control through deep reinforcement learning*, Mnih et al., 2015
3. *Deep Reinforcement Learning with Double Q-learning*, van Hasselt et al., 2015
4. *Continuous control with deep reinforcement learning*, Lillicrap et al., 2015

## Pending Tasks
- Documentation
- Implementation of A3C, Dueling DQN, Continuous DQN, NAF and CEM
