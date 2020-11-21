## Run Y2H
time=$(date "+%m%d-%H-%M-%S")
# echo $time
CUDA_VISIBLE_DEVICES=1 python main.py --runner denoiser --time $time 