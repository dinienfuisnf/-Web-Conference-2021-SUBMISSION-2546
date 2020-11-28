# COVID19-RL

This is a code description about *Reinforced Contact Tracing and*

*Intervening.* Usage and more information can be found below.

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

### Training IDRLECA:
`Train=True --python main.py --epochs 20000 --save_path save/localtime/ `

### Testing IDRLECA:
`Train=False --load_path /save/localtime/ --python main.py --save_path results/  `

### Evaluating IDRLECA on Q, I  and Score:
`file_name="results/cnt_test.txt" --python summary.py`

## Configuration
  Python3.6
  TensorFlow2.1
  Keras2.3.1






