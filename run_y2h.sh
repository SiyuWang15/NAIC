## Run Y2H
time=$(date "+%m%d-%H-%M-%S")
# echo $time
CUDA_VISIBLE_DEVICES=1,4,6,7 python main.py --runner y2h --time $time --run_mode cnn