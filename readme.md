# COVID19-RL

This is a code description about *Reinforced Contact Tracing and*

*Epidemic Intervention.* Usage and more information can be found below.

## Usage

* `main.py`

  Main file of PPO algorithm. Run this file to start an experiment.

* `layers.py`

  The file of GNN network.

* `env_new.py`

  The environment of the problem. The simulator used in the paper: 

  https://hzw77-demo.readthedocs.io/en/round2/introduction.html

* `get_args.py`

  Parameters in the model.

* `summary.py`
  
  Evaluation metrics.
## Processing

### Training IDRLECA: train.sh
`python main.py --train=True --save_name='test' --scenario='scenario1' --epochs=2
0000 --save_path='./save/' `

### Testing IDRLECA:
`python main.py --train=False ----save_name='test' --scenario='scenario1' --load_
path='./save/localtime_/' `

### Evaluating IDRLECA on Q, I  and Score:
`python summary.py --eva_text=“cnt_test.txt"`

## Configuration
  Python3.6
  TensorFlow2.1
  Keras2.3.1






