
from model import *
import torch
import sys
sys.path.append(('..'))
from utils import *
import numpy as np



def submit1(Y):
    Pilot_num = 32
    mode = 0

    # Model Construction
    FC = yp2h_estimation(1024, 4096, 2*4*256, 2)
    CNN = CNN_Estimation()
    CE2 = CNN_Estimation()

    # Load weights
    path = 'checkpoints/Pn32_0.04156.pth'
    state_dicts = torch.load(path)
    FC.load_state_dict(state_dicts['fc'])
    CNN.load_state_dict(state_dicts['cnn'])
    print("Model for CE has been loaded!")

    CE2_path = 'checkpoints/FeedbackCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CE2.load_state_dict(torch.load(CE2_path)['state_dict'])
    print("Model for CE2 has been loaded!")

    FC = torch.nn.DataParallel( FC ).cuda()  # model.module
    CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
    CE2 = torch.nn.DataParallel( CE2 ).cuda()

    # print('==========Begin Testing=========')
    FC.eval()
    CNN.eval()
    CE2.eval()

    X_bits = []
    N = 2

    with torch.no_grad():
        for i in range(2):
            Y_test = Y[i*5000 : (i+1)*5000, :]
            Ns = Y_test.shape[0]

            # 接收数据与接收导频划分
            Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_test = torch.tensor(Y_input_test).float()
            Yp_input_test = Y_input_test[:,:,0,:,:]
            Yd_input_test = Y_input_test[:,:,1,:,:]
            Yd = np.array(Yd_input_test[:, 0, :, :] + 1j * Yd_input_test[:, 1, :, :])

            # 第一层网络输入
            input1 = Yp_input_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
            input1 = input1.cuda()
            # 第一层网络输出
            output1 = FC(input1)
            # print('第一层')

            # 第二层网络输入
            output1 = output1.reshape(Ns, 2, 4, 256)
            input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)
            # 第二层网络输出
            output2 = CNN(input2)
            output2 = output2.reshape(Ns, 2, 4, 32)
            # print('第二层')

            start = output2
            for idx in range(N):

                #第三层网络输入
                H_test_padding = start.cpu()
                #### 以下 时域信道转换为频域信道 ####
                H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
                H_test_padding = H_test_padding.permute(0,2,3,1)
                H_test_padding = torch.fft(H_test_padding, 1)/20
                H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256
                #### 以上 时域信道转换为频域信道 ####
                Hf2 = H_test_padding.detach().cpu().numpy()
                Hf2 = Hf2[:, 0, :, :] + 1j * Hf2[:, 1, :, :]
                Hf2 = np.reshape(Hf2, [-1, 2, 2, 256], order='F')

                # 第三层网络输出
                X_SoftML, X_SoftMLbits, output3 = SoftMLReceiver(Yd, Hf2, SNRdb=6)
                output3 = torch.tensor(output3).float()
                X_1 = output3.cuda()
                # print('第三层')

                # 第四层网络输入
                input4 = torch.cat([X_1, Yd_input_test.cuda(), H_test_padding.cuda()], 2)
                # 第四层网络输出
                output4 = CE2(input4)
                output4 = output4.reshape(Ns, 2, 4, 32)
                # print('第四层')


                start = output4

            #### 以下 时域信道转换为频域信道 ####
            H_test_padding2 = output4.cpu().detach()
            H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
            H_test_padding2 = H_test_padding2.permute(0,2,3,1)
            H_test_padding2 = torch.fft(H_test_padding2, 1)/20
            H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous()
            #### 以上 时域信道转换为频域信道 ####

            Hf2 = H_test_padding2.numpy()
            Hf2 = Hf2[:,0,:,:] + 1j*Hf2[:,1,:,:]
            Hf2 = np.reshape(Hf2, [-1,2,2,256], order = 'F')

            # 软比特最大似然信号检测
            X_SoftML, X_SoftMLbits, _ = SoftMLReceiver(Yd, Hf2, SNRdb = 6)
            X_bits.append(X_SoftMLbits)

            print(('batch[{0}]/2 completed!'.format(i)))

    X_bits = np.concatenate(X_bits , axis = 0 )

    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('results/X_pre_1_1.bin')

def submit2(Y):
    Pilot_num = 32
    mode = 0

    # Model Construction
    FC = yp2h_estimation(1024, 4096, 2*4*256, 2)
    CNN = CNN_Estimation()
    SD = ResNet()
    CE2 = CNN_Estimation()

    # Load weights
    path = 'checkpoints/Pn32_0.04311.pth'
    state_dicts = torch.load(path)
    FC.load_state_dict(state_dicts['fc'])

    CNN = torch.nn.DataParallel(CNN).cuda()  # model.module
    CNN_path = 'checkpoints/Joint_CNN_CE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CNN.load_state_dict(torch.load(CNN_path)['state_dict'])
    print("Model for CE has been loaded!")

    SD = torch.nn.DataParallel(SD).cuda()  # model.module
    SD_path = 'checkpoints/Joint_Resnet_SD_Pilot' + str(Pilot_num) + '_mode' + str(mode) + '.pth.tar'
    SD.load_state_dict(torch.load(SD_path)['state_dict'])
    print("Model for SD has been loaded!")

    CE2_path = 'checkpoints/FeedbackCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CE2.load_state_dict(torch.load(CE2_path)['state_dict'])
    print("Model for CE2 has been loaded!")

    FC = torch.nn.DataParallel( FC ).cuda()  # model.module
    CE2 = torch.nn.DataParallel( CE2 ).cuda()

    # print('==========Begin Testing=========')
    FC.eval()
    SD.eval()
    CNN.eval()
    CE2.eval()

    X_bits = []
    N = 2

    with torch.no_grad():
        for i in range(2):
            Y_test = Y[i*5000 : (i+1)*5000, :]
            Ns = Y_test.shape[0]

            # 接收数据与接收导频划分
            Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_test = torch.tensor(Y_input_test).float()
            Yp_input_test = Y_input_test[:,:,0,:,:]
            Yd_input_test = Y_input_test[:,:,1,:,:]
            Yd = np.array(Yd_input_test[:, 0, :, :] + 1j * Yd_input_test[:, 1, :, :])

            # 第一层网络输入
            input1 = Yp_input_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
            input1 = input1.cuda()
            # 第一层网络输出
            output1 = FC(input1)
            # print('第一层')

            # 第二层网络输入
            output1 = output1.reshape(Ns, 2, 4, 256)
            input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)
            # 第二层网络输出
            output2 = CNN(input2)
            output2 = output2.reshape(Ns, 2, 4, 32)
            # print('第二层')

            start = output2
            for idx in range(N):

                #第三层网络输入
                H_test_padding = start.cpu()
                #### 以下 时域信道转换为频域信道 ####
                H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
                H_test_padding = H_test_padding.permute(0,2,3,1)
                H_test_padding = torch.fft(H_test_padding, 1)/20
                H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256
                #### 以上 时域信道转换为频域信道 ####
                input3 = torch.cat([Yp_input_test.cuda(), Yd_input_test.cuda(), H_test_padding.cuda()], 2)
                input3 = torch.reshape(input3, [Ns, 1, 16, 256])

                # 第三层网络输出
                output3 = SD(input3)
                # print('第三层')

                # 第四层网络输入
                X_1 = output3.reshape([Ns, 2, 256, 2])
                X_1 = X_1.permute(0, 3, 1, 2).contiguous()
                input4 = torch.cat([X_1, Yd_input_test.cuda(), H_test_padding.cuda()], 2)
                # 第四层网络输出
                output4 = CE2(input4)
                output4 = output4.reshape(Ns, 2, 4, 32)
                # print('第四层')

                start = output4

            #### 以下 时域信道转换为频域信道 ####
            H_test_padding2 = output4.cpu().detach()
            H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
            H_test_padding2 = H_test_padding2.permute(0,2,3,1)
            H_test_padding2 = torch.fft(H_test_padding2, 1)/20
            H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous()
            #### 以上 时域信道转换为频域信道 ####

            Hf2 = H_test_padding2.numpy()
            Hf2 = Hf2[:,0,:,:] + 1j*Hf2[:,1,:,:]
            Hf2 = np.reshape(Hf2, [-1,2,2,256], order = 'F')

            # 软比特最大似然信号检测
            X_SoftML, X_SoftMLbits, _ = SoftMLReceiver(Yd, Hf2, SNRdb = 6)
            X_bits.append(X_SoftMLbits)

            print(('batch[{0}]/2 completed!'.format(i)))

    X_bits = np.concatenate(X_bits , axis = 0 )

    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('results/X_pre_1_2.bin')

def submit3(Y):
    Pilot_num = 32
    mode = 0

    # Model Construction
    FC = yp2h_estimation(1024, 4096, 2*4*256, 2)

    CNN = CNN_Estimation()
    CE2 = CNN_Estimation()

    # Load weights
    path = 'checkpoints/Pn32_0.04156.pth'
    state_dicts = torch.load(path)
    FC.load_state_dict(state_dicts['fc'])
    CNN.load_state_dict(state_dicts['cnn'])
    print("Model for CE has been loaded!")

    SD_path = 'checkpoints/HS_SD_Pilot8'
    SD = DeepRx_port.DeepRx(SD_path, cuda_gpu=True, gpus=[0, 1])

    CE2_path = 'checkpoints/FeedbackCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CE2.load_state_dict(torch.load(CE2_path)['state_dict'])
    print("Model for CE2 has been loaded!")

    FC = torch.nn.DataParallel( FC ).cuda()  # model.module
    CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
    CE2 = torch.nn.DataParallel( CE2 ).cuda()

    print('==========Begin Testing=========')
    FC.eval()
    CNN.eval()
    SD[0].eval()
    SD[1].eval()
    CE2.eval()

    X_bits = []
    N = 1

    with torch.no_grad():
        for i in range(2):
            Y_test = Y[i*5000 : (i+1)*5000, :]
            Ns = Y_test.shape[0]

            # 接收数据与接收导频划分
            Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_test = torch.tensor(Y_input_test).float()
            Yp_input_test = Y_input_test[:,:,0,:,:]
            Yd_input_test = Y_input_test[:,:,1,:,:]
            Yd = np.array(Yd_input_test[:, 0, :, :] + 1j * Yd_input_test[:, 1, :, :])

            # 第一层网络输入
            input1 = Yp_input_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
            input1 = input1.cuda()
            # 第一层网络输出
            output1 = FC(input1)
            # print('第一层')

            # 第二层网络输入
            output1 = output1.reshape(Ns, 2, 4, 256)
            input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)
            # 第二层网络输出
            output2 = CNN(input2)
            output2 = output2.reshape(Ns, 2, 4, 32)
            # print('第二层')

            start = output2
            for idx in range(N):

                #第三层网络输入
                H_test_padding = start.cpu()
                #### 以下 时域信道转换为频域信道 ####
                H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
                H_test_padding = H_test_padding.permute(0,2,3,1)
                H_test_padding = torch.fft(H_test_padding, 1)/20
                H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256
                #### 以上 时域信道转换为频域信道 ####


                # 第三层网络输出
                output3 = DeepRx_port.Pred_DeepRX(SD, torch.reshape(H_test_padding.cuda(), [-1, 2, 2, 2, 256]),
                                             Yd_input_test.cuda())
                # print('第三层')

                # 第四层网络输入
                X_1 = output3.reshape([Ns, 2, 256, 2])
                X_1 = X_1.permute(0, 3, 1, 2).contiguous()
                input4 = torch.cat([X_1, Yd_input_test.cuda(), H_test_padding.cuda()], 2)
                # 第四层网络输出
                output4 = CE2(input4)
                output4 = output4.reshape(Ns, 2, 4, 32)
                # print('第四层')

                #第五层网络输入预处理
                start = output4

            #### 以下 时域信道转换为频域信道 ####
            H_test_padding2 = output4.cpu().detach()
            H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
            H_test_padding2 = H_test_padding2.permute(0,2,3,1)
            H_test_padding2 = torch.fft(H_test_padding2, 1)/20
            H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous()
            #### 以上 时域信道转换为频域信道 ####

            Hf2 = H_test_padding2.numpy()
            Hf2 = Hf2[:,0,:,:] + 1j*Hf2[:,1,:,:]
            Hf2 = np.reshape(Hf2, [-1,2,2,256], order = 'F')

            # 软比特最大似然信号检测
            X_SoftML, X_SoftMLbits, _ = SoftMLReceiver(Yd, Hf2, SNRdb = 6)
            X_bits.append(X_SoftMLbits)

            print(('batch{0} completed!'.format(i)))

    X_bits = np.concatenate(X_bits , axis = 0 )

    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('results/X_pre_1_3.bin')

def submit4(Y):
    Pilot_num = 32
    mode = 0

    # Model Construction
    FC = yp2h_estimation(1024, 4096, 2*4*256, 2)

    CNN = CNN_Estimation()
    CE2 = CNN_Estimation()

    # Load weights
    path = 'checkpoints/Pn32_0.04156.pth'
    state_dicts = torch.load(path)
    FC.load_state_dict(state_dicts['fc'])
    CNN.load_state_dict(state_dicts['cnn'])
    print("Model for CE has been loaded!")

    SD_path = 'checkpoints/HS_SD_Pilot8'
    SD = DeepRx_port.DeepRx(SD_path, cuda_gpu=True, gpus=[0, 1])

    CE2_path = 'checkpoints/JointJoint_FeedbackCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CE2.load_state_dict(torch.load(CE2_path)['state_dict'])
    print("Model for CE2 has been loaded!")

    FC = torch.nn.DataParallel( FC ).cuda()  # model.module
    CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
    CE2 = torch.nn.DataParallel( CE2 ).cuda()

    print('==========Begin Testing=========')
    FC.eval()
    CNN.eval()
    SD[0].eval()
    SD[1].eval()
    CE2.eval()

    X_bits = []
    N = 1

    with torch.no_grad():
        for i in range(2):
            Y_test = Y[i*5000 : (i+1)*5000, :]
            Ns = Y_test.shape[0]

            # 接收数据与接收导频划分
            Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_test = torch.tensor(Y_input_test).float()
            Yp_input_test = Y_input_test[:,:,0,:,:]
            Yd_input_test = Y_input_test[:,:,1,:,:]
            Yd = np.array(Yd_input_test[:, 0, :, :] + 1j * Yd_input_test[:, 1, :, :])

            # 第一层网络输入
            input1 = Yp_input_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
            input1 = input1.cuda()
            # 第一层网络输出
            output1 = FC(input1)
#             print('第一层')

            # 第二层网络输入
            output1 = output1.reshape(Ns, 2, 4, 256)
            input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)
            # 第二层网络输出
            output2 = CNN(input2)
            output2 = output2.reshape(Ns, 2, 4, 32)
            # print('第二层')

            start = output2
            for idx in range(N):

                #第三层网络输入
                H_test_padding = start.cpu()
                #### 以下 时域信道转换为频域信道 ####
                H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
                H_test_padding = H_test_padding.permute(0,2,3,1)
                H_test_padding = torch.fft(H_test_padding, 1)/20
                H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256
                #### 以上 时域信道转换为频域信道 ####


                # 第三层网络输出
                output3 = DeepRx_port.Pred_DeepRX(SD, torch.reshape(H_test_padding.cuda(), [-1, 2, 2, 2, 256]),
                                             Yd_input_test.cuda())
                # print('第三层')

                # 第四层网络输入
                X_1 = output3.reshape([Ns, 2, 256, 2])
                X_1 = X_1.permute(0, 3, 1, 2).contiguous()
                input4 = torch.cat([X_1, Yd_input_test.cuda(), H_test_padding.cuda()], 2)
                # 第四层网络输出
                output4 = CE2(input4)
                output4 = output4.reshape(Ns, 2, 4, 32)
                # print('第四层')

                #第五层网络输入预处理
                start = output4

            #### 以下 时域信道转换为频域信道 ####
            H_test_padding2 = output4.cpu().detach()
            H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
            H_test_padding2 = H_test_padding2.permute(0,2,3,1)
            H_test_padding2 = torch.fft(H_test_padding2, 1)/20
            H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous()
            #### 以上 时域信道转换为频域信道 ####

            Hf2 = H_test_padding2.numpy()
            Hf2 = Hf2[:,0,:,:] + 1j*Hf2[:,1,:,:]
            Hf2 = np.reshape(Hf2, [-1,2,2,256], order = 'F')

            # 软比特最大似然信号检测
            X_SoftML, X_SoftMLbits, _ = SoftMLReceiver(Yd, Hf2, SNRdb = 6)
            X_bits.append(X_SoftMLbits)

            print(('batch[{0}]/2 completed!'.format(i)))

    X_bits = np.concatenate(X_bits , axis = 0 )

    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('results/X_pre_1_4.bin')

def submit5(Y):
    Pilot_num = 32
    mode = 0

    # Model Construction
    FC = yp2h_estimation(1024, 4096, 2*4*256, 2)

    CNN = CNN_Estimation()
    CE2 = CNN_Estimation()

    # Load weights
    path = 'checkpoints/Pn32_0.04156.pth'
    state_dicts = torch.load(path)
    FC.load_state_dict(state_dicts['fc'])
    CNN.load_state_dict(state_dicts['cnn'])
    print("Model for CE has been loaded!")

    SD_path = 'checkpoints/HS_SD_Pilot8'
    SD = DeepRx_port.DeepRx(SD_path, cuda_gpu=True, gpus=[0, 1])

    CE2_path = 'checkpoints/JointCESD_FeedbackCE_Pilot'+ str(Pilot_num)+'_mode' + str(mode) + '.pth.tar'
    CE2.load_state_dict(torch.load(CE2_path)['state_dict'])
    print("Model for CE2 has been loaded!")

    FC = torch.nn.DataParallel( FC ).cuda()  # model.module
    CNN = torch.nn.DataParallel( CNN ).cuda()  # model.module
    CE2 = torch.nn.DataParallel( CE2 ).cuda()

    print('==========Begin Testing=========')
    FC.eval()
    CNN.eval()
    SD[0].eval()
    SD[1].eval()
    CE2.eval()

    X_bits = []
    N = 1

    with torch.no_grad():
        for i in range(2):
            Y_test = Y[i*5000 : (i+1)*5000, :]
            Ns = Y_test.shape[0]

            # 接收数据与接收导频划分
            Y_input_test = np.reshape(Y_test, [Ns, 2, 2, 2, 256], order='F')
            Y_input_test = torch.tensor(Y_input_test).float()
            Yp_input_test = Y_input_test[:,:,0,:,:]
            Yd_input_test = Y_input_test[:,:,1,:,:]
            Yd = np.array(Yd_input_test[:, 0, :, :] + 1j * Yd_input_test[:, 1, :, :])

            # 第一层网络输入
            input1 = Yp_input_test.reshape(Ns, 2*2*256) # 取出接收导频信号，实部虚部*2*256
            input1 = input1.cuda()
            # 第一层网络输出
            output1 = FC(input1)
            # print('第一层')

            # 第二层网络输入
            output1 = output1.reshape(Ns, 2, 4, 256)
            input2 = torch.cat([Yd_input_test.cuda(), Yp_input_test.cuda(), output1], 2)
            # 第二层网络输出
            output2 = CNN(input2)
            output2 = output2.reshape(Ns, 2, 4, 32)
            # print('第二层')

            start = output2
            for idx in range(N):

                #第三层网络输入
                H_test_padding = start.cpu()
                #### 以下 时域信道转换为频域信道 ####
                H_test_padding = torch.cat([H_test_padding, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
                H_test_padding = H_test_padding.permute(0,2,3,1)
                H_test_padding = torch.fft(H_test_padding, 1)/20
                H_test_padding = H_test_padding.permute(0,3,1,2).contiguous() #df1为对应的频域形式，维度为Batch*2*4*256
                #### 以上 时域信道转换为频域信道 ####


                # 第三层网络输出
                output3 = DeepRx_port.Pred_DeepRX(SD, torch.reshape(H_test_padding.cuda(), [-1, 2, 2, 2, 256]),
                                             Yd_input_test.cuda())
                # print('第三层')

                # 第四层网络输入
                X_1 = output3.reshape([Ns, 2, 256, 2])
                X_1 = X_1.permute(0, 3, 1, 2).contiguous()
                input4 = torch.cat([X_1, Yd_input_test.cuda(), H_test_padding.cuda()], 2)
                # 第四层网络输出
                output4 = CE2(input4)
                output4 = output4.reshape(Ns, 2, 4, 32)
                # print('第四层')

                #第五层网络输入预处理
                start = output4

            #### 以下 时域信道转换为频域信道 ####
            H_test_padding2 = output4.cpu().detach()
            H_test_padding2 = torch.cat([H_test_padding2, torch.zeros(Ns,2,4,256-32, requires_grad=True)],3)
            H_test_padding2 = H_test_padding2.permute(0,2,3,1)
            H_test_padding2 = torch.fft(H_test_padding2, 1)/20
            H_test_padding2 = H_test_padding2.permute(0,3,1,2).contiguous()
            #### 以上 时域信道转换为频域信道 ####

            Hf2 = H_test_padding2.numpy()
            Hf2 = Hf2[:,0,:,:] + 1j*Hf2[:,1,:,:]
            Hf2 = np.reshape(Hf2, [-1,2,2,256], order = 'F')

            # 软比特最大似然信号检测
            X_SoftML, X_SoftMLbits, _ = SoftMLReceiver(Yd, Hf2, SNRdb = 6)
            X_bits.append(X_SoftMLbits)

            print(('batch[{0}]/2 completed!'.format(i)))

    X_bits = np.concatenate(X_bits , axis = 0 )

    X_1 = np.array(np.floor(X_bits + 0.5), dtype=np.bool)
    X_1.tofile('results/X_pre_1_5.bin')