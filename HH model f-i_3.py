from brian2 import *
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
'HH模型+F-I曲线。在2的基础上更新'
'使用Matlab的参数配置，变量为3个电导的系数，并不改变动力学方程的系数和电压依赖性等'
'由于以后需要树突，因此吧面积项附加上来了'
'excitable全是type-I，pacemaker是Tipe-II，去对应的文件里面拿参数就行'
'系数同步之后就可以搜参了'
# 设置参数
area = 20000*umetre**2
Cm = 1*ufarad*cm**-2 * area
El = -75.6*mV
EK = -77*mV
ENa = 50*mV
# El = -10.6*mV 这是matlab里的值，但是matlab中，微分方程里面都额外减少了65
# EK = -12*mV
# ENa = 115*mV
g_na = 120*msiemens*cm**-2 * area
g_kd = 36*msiemens*cm**-2 * area
gl = 3e-4*siemens*cm**-2 * area   #matlab写的0.3，但matlab中3电导同单位，这里不是同单位，要注意改变
'系数，Type-I，电流2.5-2.7出现'
g_NaL=1.2
g_KL=1.2378
g_LL=0.9157
#Type-II
# g_NaL=1.2419
# g_KL=0.7921
# g_LL=1.1983


# HH模型方程 （添加了系数，其余不变）
eqs = '''
dv/dt = (g_NaL*g_na*m**3*h*(ENa-v) + g_KL*g_kd*n**4*(EK-v) + g_LL*gl*(El-v) + I)/Cm : volt
dm/dt = alpha_m*(1-m) - beta_m*m : 1
dh/dt = alpha_h*(1-h) - beta_h*h : 1
dn/dt = alpha_n*(1-n) - beta_n*n : 1
alpha_m = 0.1/mV*10*mV/exprel(-(v+40*mV)/(10*mV))/ms : Hz
beta_m = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
alpha_h = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
beta_h = 1/(1+exp(-(v+35*mV)/(10*mV)))/ms : Hz
alpha_n = 0.01/mV*10*mV/exprel(-(v+55*mV)/(10*mV))/ms : Hz
beta_n = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
I : amp
'''

def simulate_for_current(Input):
    # 重置默认时钟
    defaultclock.dt = 0.01*ms# 设置时间步长为0.01毫秒,如果电流步长太小，可能需要也变小，或者method换成rk4（可变步长求解器

    # 创建神经元群
    neuron = NeuronGroup(1, eqs, method='exponential_euler',
                         threshold='v > 30*mV',
                         reset='',  #不能设置，会影响动力学
                         refractory=2*ms)

    # 初始化变量
    neuron.v = El
    neuron.m = 0.05
    neuron.h = 0.6
    neuron.n = 0.32
    # neuron.m = 'alpha_m/(alpha_m+beta_m)'
    # neuron.h = 'alpha_h/(alpha_h+beta_h)'
    # neuron.n = 'alpha_n/(alpha_n+beta_n)'

    # 创建监视器
    state_monitor = StateMonitor(neuron, 'v', record=True)
    spike_monitor = SpikeMonitor(neuron)

    # 创建网络并添加所有组件
    net = Network(neuron, state_monitor, spike_monitor)

    # 先运行一段时间让模型达到稳态
    neuron.I = 0 * nA
    net.run(200 * ms)

    # 设置输入电流
    neuron.I = Input

    # 运行模拟
    net.run(1000 * ms)

    # 计算发放频率
    if len(spike_monitor.t) > 1:
        ISIs = np.diff(spike_monitor.t)
        mean_ISI = np.mean(ISIs)
        firing_rate = 1 / mean_ISI
        #firing_rate = len(spike_mon.t) / (spike_mon.t[-1] - spike_mon.t[0])  # 另一个方法
    else:
        firing_rate = 0

    # 返回可序列化的数据,因为并行计算，而多进程环境中Brian2的对象无法直接传递（有单位），必须变成纯数据

    return Input/nA, firing_rate/Hz, state_monitor.t/ms, state_monitor.v[0]/mV

if __name__ == '__main__':
    I_range = np.arange(2, 10, 0.1) * nA  # 含头不含尾
    #2-3是typeII

    # 使用multiprocessing进行并行处理
    with Pool() as pool:
        results = pool.map(simulate_for_current, I_range)

    # 处理结果
    currents, firing_rates, all_times, all_voltages = zip(*results)

    # 绘制f-I曲线
    plt.figure(figsize=(10, 6))
    plt.plot(currents, firing_rates, 'o-')
    plt.xlabel('Input Current (nA)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('f-I Curve')
    plt.show()

    # 显示所有的膜电位图像
    for i, (times, voltages) in enumerate(zip(all_times, all_voltages)):
        plt.figure(figsize=(10, 6))
        plt.plot(times, voltages)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane potential (mV)')
        plt.title(f'Membrane potential for I = {currents[i]:.3g} nA')
        plt.show()