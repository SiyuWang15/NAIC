time=$(date "+%m%d-%H-%M-%S")

# # train FC coarse channel estimation
# CUDA_VISIBLE_DEVICES=0 python main.py --runner y2h --time $time --run_mode fc

# # train CNN finer channel estimation 
# CUDA_VISIBLE_DEVICES=0 python main.py --runner y2h --time $time --run_mode cnn

# # train CNN finer channel estimation via Exponential Moving Averaage trick
# CUDA_VISIBLE_DEVICES=0 python main.py --runner ema --time $time --run_mode cnn

# # validate and test fc + cnn channel estimation and SoftMLReceiver 
# # This is not what we finally do to obtain our submit files. 
# CUDA_VISIBLE_DEVICES=6 python main.py --runner validation --Pn 8 --time $time

# # train SD + CE following FC + CNN channel estimation
# # For this code, only Pilotnum 8 can be trained in this mode.
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --runner sdce --Pn 8 --time $time

# ======================================================================================================
# To generate final submitted .bin file, please run this line

cd Pn32_Final_Submit
CUDA_VISIBLE_DEVICES=6,7 python main.py 
cd ..
cd Pn8_Final_Submit
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
cd ..
mkdir Submit
cp ./Pn32_Final_Submit/results/X_pre_2.bin ./Submit/
cp ./Pn8_Final_Submit/results/X_pre_1.bin ./Submit/