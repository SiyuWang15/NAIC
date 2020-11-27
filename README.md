# NAIC
Codes for AI wireless communication contest Stage2.

Out codes are built mainly on pytorch1.6.0. To quickly start, run

`pip install -r requirements.txt`

All commands can be found in run.sh.
## Generate submitted files

To generate final submitted files, please uncomment line 34-42 in *run.sh*.

## Train our model
To train our model, you may need several steps:

Train FC coarse channel estimation using config file in [./configs/y2h_config_fc.yml], run

`python main.py --runner y2h --time default --run_mode fc`

The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/FC/default ]

Train CNN finer channel estimation using config file in [./configs/y2h_config_cnn.yml], run

`python main.py --runner y2h --time default --run_mode cnn`

The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/default ]

Train CNN finer channel estimation via Exponential Moving Averaage trick using config file in [./configs/y2h_config_ema.yml] (You may not need this), run

`python main.py --runner ema --time default --run_mode cnn`

The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/EMA/default ]

Validate and test fc + cnn channel estimation and use SoftMLReceiver to predict X, run

`python main.py --runner validation --Pn 8 --time default`

The results and logging are save in [ ./workspace/validation/mode_0_Pn_8/default ]

Train SD + CE following FC + CNN channel estimation using config file in [./configs/y2h_config_sdce.yml], run

`python main.py --runner sdce --Pn 8 --time default`

The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/FC/default ]
