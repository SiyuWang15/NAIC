## Run Y2H
time=$(date "+%m%d-%H-%M-%S")
# echo $time
CUDA_VISIBLE_DEVICES=4 python main.py --runner y2h --time $time --run_mode cnn