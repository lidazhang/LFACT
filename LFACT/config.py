from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.0005#0.0005
  max_grad_norm = 5
  num_layers = 2
  num_steps = 50 # #20
  hidden_size = 128#200 #stock num
  stocks = 1#22
  classes = 256
  max_epoch = 4
  
  max_max_epoch = 150 #13
  predict_num = 5
  ponder_time_penalty = 0.006
  min_act_step = 1.0
  
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 32
  vocab_size = 10000
  max_computation = 3
  epsilon = 0.01
  use_lstm = False
  

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 0.0005
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 300
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  max_computation = 50
  epsilon = 0.01
  ponder_time_penalty = 0.01
  use_lstm = False

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.0005
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  max_computation = 50
  epsilon = 0.01
  ponder_time_penalty = 0.01
  use_lstm = False

class TitanXConfig(object):
  """For Titan X -- Faster Training"""
  init_scale = 0.04
  learning_rate = 0.0005
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.10
  batch_size = 64
  vocab_size = 10000
  max_computation = 50
  epsilon = 0.01
  ponder_time_penalty = 0.01
  use_lstm = False

