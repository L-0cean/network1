from brian2 import *
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
'直接尝试更改' '去除面积参数'
'相比于HH模型，胞体中Na乘数、动力学都变了，K变化很大，还多了ih电流'

# 参数
Cs = 1.5*uF/cm**2
Cd = 1.5*uF/cm**2

El = -77*mV
EK = -95*mV
ENa = 45*mV
Eih = -20*mV

g_Na = 40*msiemens/cm**2
g_ks = 8.75 * msiemens/cm**2
g_kd = 12* msiemens /cm**2
g_H = 0.03*msiemens/cm**2
gl= 0.032 * msiemens/cm**2
'系数'
g_NaL=1
g_KL=1
g_LL=1
# 其他的参数
R=0.75 * kohm * cm**2 #耦合电阻
tau_ih = 100*ms
tau_nd = 15*ms

'突触噪声'
'目前都置零'
I_syn_s_value=0 *uA/cm**2
I_syn_d_value=0 *uA/cm**2

I_range = np.arange(0.2,0.3,0.01) *uA/cm**2

'模型方程'
'输入电流给树突了'
eqs = '''
# Soma compartment
dv/dt = (g_NaL*g_Na*m_inf*h*(ENa-v) + g_KL*g_ks*(1-h)*(EK-v) + g_LL*gl*(El-v) 
        + g_H*ih*(Eih-v) +(vd-v)/R + I_syn_s)/Cs : volt

# Dendrite compartment
dvd/dt = (gl*(El-vd) + g_kd*nd*(EK-vd) + (v-vd)/R  + I + I_syn_d)/Cd : volt

# Sodium activation
m_inf = 1/(1 + exp((v/mV + 40)/-3)) : 1
dh/dt = (h_inf - h)/tau_h : 1
h_inf = 1/(1 + exp((v/mV + 40)/3)) : 1
tau_h = (295.4/(4*(v/mV + 50)**2 + 400) + 0.012)*ms : second

# I_H activation
dih/dt = (ih_inf - ih)/tau_ih : 1
ih_inf = 1/(1 + exp((v/mV + 80)/3)) : 1

# Slow K+ activation
dnd/dt = (nd_inf - nd)/tau_nd : 1
nd_inf = 1/(1 + exp((vd/mV + 35)/-3)) : 1

# Input current
I : amp/meter**2

I_syn_s : amp/meter**2
I_syn_d : amp/meter**2

'''



def simulate_for_current(Input):
    # 重置默认时钟
    defaultclock.dt = 0.001*ms# 设置时间步长为0.01毫秒,如果电流步长太小，可能需要也变小，或者method换成rk4（可变步长求解器

    # 创建神经元群
    neuron = NeuronGroup(1, eqs, method='exponential_euler',
                         threshold='v > 0*mV',
                         reset='',  #不能设置，会影响动力学
                         refractory=1*ms)

    # 初始化变量,必须这样赋值给神经元组，否则算是二次定义
    # neuron.v = El
    # neuron.vd = El
    # neuron.h = 1 / (1 + np.exp((El / mV + 40) / 3))
    # neuron.ih = 1 / (1 + np.exp((El / mV + 80) / 3))
    # neuron.nd = 1 / (1 + np.exp((El / mV + 35) / -3))
    E_i= -70*mV
    neuron.v =  E_i
    neuron.vd =  E_i
    neuron.h = 1 / (1 + np.exp(( E_i / mV + 40) / 3))
    neuron.ih = 1 / (1 + np.exp(( E_i / mV + 80) / 3))
    neuron.nd = 1 / (1 + np.exp(( E_i / mV + 35) / -3))

    neuron.I_syn_s = I_syn_s_value
    neuron.I_syn_d = I_syn_d_value

    # 创建监视器
    state_monitor = StateMonitor(neuron, ['v','vd'], record=True)
    spike_monitor = SpikeMonitor(neuron)

    # 创建网络并添加所有组件
    net = Network(neuron, state_monitor, spike_monitor)

    # 先运行一段时间让模型达到稳态
    neuron.I = 0 *uA/cm**2
    net.run(300 * ms)

    # 设置输入电流
    neuron.I = Input

    # 运行模拟
    time_total = 1000 * ms
    net.run(time_total)

    neuron.I = 0 *uA/cm**2
    net.run(300 * ms)


    # 基于ISI平均间隔时间计算发放频率
    if len(spike_monitor.t) > 1:
        ISIs = np.diff(spike_monitor.t)
        mean_ISI = np.mean(ISIs)
        firing_rate_ISI = 1 / mean_ISI
        #firing_rate_ISI = len(spike_mon.t) / (spike_mon.t[-1] - spike_mon.t[0])  # 另一个方法
    else:
        firing_rate_ISI = 0

    # 基于总时间计算发放频率
    firing_rate_total = len(spike_monitor.t) / time_total

    # 返回可序列化的数据,因为并行计算，而多进程环境中Brian2的对象无法直接传递（有单位），必须变成纯数据

    return Input/(uA/cm**2), firing_rate_ISI/Hz,  firing_rate_total/Hz, state_monitor.t/ms, state_monitor.v[0]/mV

if __name__ == '__main__':
    #I_range = np.arange(4.8, 5.5, 0.05) * nA  # 放在前面了，便于改数

    # 使用multiprocessing进行并行处理
    with Pool() as pool:
        results = pool.map(simulate_for_current, I_range)

    # 处理结果
    currents, firing_rate_ISI, firing_rate_total,all_times, all_voltages = zip(*results)

    # 绘制f-I曲线（两种方法）
    plt.figure(figsize=(12, 6))
    plt.plot(currents, firing_rate_ISI, 'o-', label='ISI based')
    plt.plot(currents, firing_rate_total, 's-', label='Total time based')
    plt.xlabel('Input Current (*uA/cm**2)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('f-I Curve')
    plt.legend()
    plt.show()

    # 显示所有的膜电位图像
    for i, (times, voltages) in enumerate(zip(all_times, all_voltages)):
        plt.figure(figsize=(10, 6))
        plt.plot(times, voltages)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane potential (mV)')
        plt.title(f'Membrane potential for I = {currents[i]:.3g} *uA/cm**2')
        plt.show()