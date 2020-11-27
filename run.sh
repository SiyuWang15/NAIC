# time=$(date "+%m%d-%H-%M-%S")

# # train FC coarse channel estimation
# # use config file in configs/y2h_config_fc.yml
# # The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/FC/default ]

# CUDA_VISIBLE_DEVICES=0 python main.py --runner y2h --time default --run_mode fc

# # train CNN finer channel estimation 
# # use config file in configs/y2h_config_cnn.yml
# # The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/default ]

# CUDA_VISIBLE_DEVICES=1 python main.py --runner y2h --time default --run_mode cnn

# # train CNN finer channel estimation via Exponential Moving Averaage trick.
# # use config file in configs/y2h_config_ema.yml

# CUDA_VISIBLE_DEVICES=0 python main.py --runner ema --time default --run_mode cnn

# # validate and test fc + cnn channel estimation and use SoftMLReceiver to predict X
# # This is not what we finally do to obtain our submit files. 

# CUDA_VISIBLE_DEVICES=0 python main.py --runner validation --Pn 8 --time default

# # train SD + CE following FC + CNN channel estimation
# # use config file in configs/y2h_config_sdce.yml
# # The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/FC/default ]

# CUDA_VISIBLE_DEVICES=0,1 python main.py --runner sdce --Pn 8 --time SDCE

# ======================================================================================================
# To generate final submitted .bin file, please run this line

cd Pn32_Final_Submit
CUDA_VISIBLE_DEVICES=0,1 python main.py 
cd ..
cd Pn8_Final_Submit
CUDA_VISIBLE_DEVICES=0,1 python main.py
cd ..
mkdir Submit
cp ./Pn8_Final_Submit/X_pre_2.bin ./Submit/
cp ./Pn32_Final_Submit/X_pre_1.bin ./Submit/