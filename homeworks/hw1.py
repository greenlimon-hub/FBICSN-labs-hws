import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


#Перевод частоты в циклическую
def f2w(f):
    return 2.0*pi*f

def Z1(f, C1):
    return 2.0/(1j*f2w(f)*C1)

def Z2 (f, C2):
    return 1.0/(1j*f2w(f)*C2 + G)

def Z3(f, L):
    return 1.0j*f2w(f)*L

def Gam(f, L, C1, C2):
    ZY = (Z2(f, C2)+Z3(f, L))/Z1(f, C1)
    return 2.0 * np.arcsinh(np.sqrt(ZY))

def Zw_func(f, L, C1, C2):
    return np.sqrt((Z1(f, C1)**2*(Z2(f, C2)+Z3(f, L)))/(2*Z1(f, C1)+Z2(f, C2)+Z3(f, L)))


#Производная гармонического сигнала
def d_harm_signal(t):
    return 2.0*pi*fc*sin(2.0*pi*fc*t)

#Производная гармонического сигнала с плавным нарастанием
def d_harm_imp(t):
    if t > T_start and t < T_stop:
        return 2.0*pi*fc*sin(2.0*pi*fc*t)
    else:
        return 0

#Производная НЧ импульса
def d_LFimpulse(t):
    if T_start <= t < T_start + fwFront:
        return 1 / fwFront
    elif T_stop - bwFront <= t < T_stop:
        return -1/ bwFront
    else:
        return 0

#Производная НЧ + ВЧ импульса
def d_TWOimpulses(t):
    if T_start <= t < T_start + fwFront:
        return 1 / fwFront
    elif T_start + fwFront <= t < T_stop - bwFront:
        return 0.1 * 2.0*pi*fc*cos(2.0*pi*fc*t)
    elif T_stop - bwFront <= t < T_stop:
        return -1/ bwFront
    else:
        return 0

#Производная широкополосного сигнала
def d_BBsignal(t):
    ddt = 1 / (fh*0.95 - fl*1.2)
    fc1 = (fh*0.95 + fl*1.2) / 2
    T = Tc
    # Гауссовская огибающая и несущая
    gaussian = np.exp(-( (T/2 - t)**2 ) / (2 * ddt**2))
    carrier = np.sin(2 * np.pi * fc1 * t)
    signal = gaussian * carrier
    # Производная огибающей (du/dt)
    dgaussian_dt = gaussian * ( (T/2 - t) / ddt**2 )
    # Производная несущей (dv/dt)
    dcarrier_dt = 2 * np.pi * fc1 * np.cos(2 * np.pi * fc1 * t)
    # Полная производная сигнала: ds/dt = du/dt * v + u * dv/dt
    return dgaussian_dt * carrier + gaussian * dcarrier_dt



global nvar, fc, Tc, fl, fh, Vinp, time, Vout, fft_freq, sp_inp, sp_out, Vs, T_start, T_stop, fwFront, bwFront, G


nvar = 2

fc = float(input('Частота сигнала '))
Tc = 4

T_start = Tc*0.1
T_stop=0.3*Tc

fwFront = bwFront = 0.01*Tc

fl = nvar
fh = 10 * (nvar + 1)
f0 = (fl + fh) * 0.5
Z0 = 10 * nvar

type_signal = int(input('Тип сигнала\n1 - ВЧ сигнал\n2 - ВЧ импульс\n3 - НЧ импульс\n4 - ВЧ+НЧ импульс\n5 - ШП '
                        'импульс'))
Nc = int(input('Число ячеек в ЛП '))

L = (sqrt(Z0**2*f2w(f0)**2*(2*f2w(fh)**2-f2w(fl)**2-f2w(f0)**2)/
    ((f2w(fh)**2-f2w(fl)**2)**2*(f2w(f0)**2-f2w(fl)**2))))
C1 = 2.0 / L / (f2w(fh)**2 - f2w(fl)**2)
C2 = 1.0 / (f2w(fl)**2 * L)
G = 0

print('Параметры отдельной ячейки ЛП:')
print('C1 = {0: f}\nC2 = {1: f}\nL = {2: f}'.format(C1, C2, L))

npp = 10            #Количество точек на период гармонического сигнала
dt = 1/(fc*npp)     #Шаг по времени
num = int(Tc / dt)  #Количество временных отсчетов

freq = np.linspace(0.8*fl, fh*1.2, num)

#Задание производной сигнала возбуждения ЛП
if type_signal == 1:
    d_signal = d_harm_signal
elif type_signal == 2:
    d_signal = d_harm_imp
elif type_signal == 3:
    d_signal = d_LFimpulse
elif type_signal == 4:
    d_signal = d_TWOimpulses
elif type_signal == 5:
    d_signal = d_BBsignal

A0 = 1 #Амплитуда сигнала слева
AN = 0 #Амплитуда сигнала справа
K0 = KN = 1 #Коэффициенты при нагрузочных сопротивлениях
# K0 = 2
# KN = 5

#Количество итераций для решения уравнений возбуждения
dpp = 20
print('dpp = {0: d}'.format(dpp))

aU = [0] * Nc     #Массив напряжений на емкости C2
dU = [0] * Nc     #Массив производных напряжений на емкости C2
aV = [0] * (Nc+1) #Массив напряжений на емкости C1
dV = [0] * (Nc+1) #Массив производных напряжений на емкости C1

Vinp = [0] * num  #Массив входных напряжений
Vout = [0] * num  #Массив выходных напряжений
time = [0] * num  #Массив временных отсчетов

Vs = [0] * npp    #Массив напряжений на C1 вдоль ЛП на одном периоде сигнала
for i in range(npp): Vs[i] = [0] * (Nc+1)

#Решение уравнений возбуждения ЛП
for it in range(num):
    time[it] = dt * it
    for i in range(dpp):
        dV[0] += (1.0/(L*C1)*(aV[1]-aV[0]+aU[0])+1.0/(Z0*K0*C1)*(A0*d_signal(time[it])-dV[0]))*dt/dpp
        for ic in range (Nc):
            dU[ic] += (1.0/(L*C2)*(aV[ic]-aV[ic+1]-aU[ic])-G/C2*dU[ic])*dt/dpp
            if ic == 0: continue
            dV[ic] += (0.5/(L*C1)*(aV[ic-1]-2.0*aV[ic]+aV[ic+1]+aU[ic]-aU[ic-1]))*dt/dpp
        dV[Nc] += (1.0/(L*C1)*(aV[Nc-1]-aV[Nc]-aU[Nc-1])+1.0/(Z0*KN*C1)*(AN*d_signal(time[it])-dV[Nc]))*dt/dpp

        for ic in range(Nc):
            aV[ic] += dV[ic]*dt/dpp
            aU[ic] += dU[ic]*dt/dpp
        aV[Nc] += dV[Nc]*dt/dpp

    if num-it <= npp:
        for ic in range(Nc+1):
            Vs[it-(num-npp)][ic] = aV[ic]

    Vinp[it] = aV[0]
    Vout[it] = aV[Nc]
    if it % 100 == 0:
        print('{0: 7.3f} {1: 7.3f} {2: 7.3f} '.format(time[it], Vinp[it], Vout[it]))


#Расчет спектра входного и выходного сигалов
spectr_inp = np.fft.fft(Vinp)
spectr_out = np.fft.fft(Vout)
fft_freq = np.fft.fftfreq(num, Tc/num)

# plt.figure()
# plt.plot(time, Vinp, time, Vout)
# plt.show()
#
# plt.figure()
sp_inp = np.hypot(spectr_inp.real, spectr_inp.imag)/num*2
sp_out = np.hypot(spectr_out.real, spectr_out.imag)/num*2
# plt.plot(fft_freq[0:num//2], sp_inp[0:num//2], label='$V_{inp}$')
# plt.plot(fft_freq[0:num//2], sp_out[0:num//2], label='$V_{out}$')
# plt.legend(loc='best')
# plt.show()

# plt.figure()
# cells = np.linspace(0, Nc, Nc+1)
# z_spl = np.linspace(0, Nc, (Nc+1)*10)
# for i in range(npp):
#     spl = make_interp_spline(cells, Vs[i], k=3)
#     plt.plot(z_spl, spl(z_spl), label="t = {0: .3f} Ñ ".format(time[num-npp+i]), lw=1)
# plt.legend(loc='best')
# plt.show()


Gama = Gam(freq, L, C1, C2)
Zw = Zw_func(freq, L, C1, C2)
dF = (Gam(freq+0.01, L, C1, C2).imag-Gam(freq-0.01, L, C1, C2).imag) / 0.02


# Построение графиков
def drawAllGraph(separate=False):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    fig.tight_layout(pad=3.0)
    ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()
    drawGraph(fig, ax1, 0)
    drawGraph(fig, ax2, 1)
    drawGraph(fig, ax3, 2)
    drawGraph(fig, ax4, 3)
    drawGraph(fig, ax5, 4)
    drawGraph(fig, ax6, 5)
    plt.subplots_adjust(hspace=0.42)
    plt.show()



def drawGraph(fig, ax, ng, title=True):
    if ng == 0:
        ax.plot(time, Vinp, lw=1, color='#539caf', alpha=1)
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(min(Vinp) * 1.1, max(Vinp) * 1.1)
        if title: ax.set_title("(а) - $U_{вх}(t)$", fontsize=10)
        ax.set_xlabel(r"Время t, с", fontsize=10)
        ax.set_ylabel("$U_{вх}$, В", fontsize=10)

    elif ng == 1:
        ax.plot(time, Vout, lw=1, color='#539caf', alpha=1)
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(min(Vout) * 1.1, max(Vout) * 1.1)
        if title: ax.set_title("(б) - $U_{вых}(t)$", fontsize=10)
        ax.set_xlabel(r"Время t, с", fontsize=10)
        ax.set_ylabel("$U_{вых}$, В", fontsize=10)

    elif ng == 2:
        ax.plot(fft_freq[0:num//2], sp_inp[0:num//2], lw=1, color='#539caf', alpha=1)
        ax.set_xlim(0, max(fh, fc) * 1.05)
        # ax.set_ylim(min(sp_inp[0:num//2] * 1.1, max(sp_inp[0:num//2]) * 1.05))
        if title: ax.set_title("(в) - $U_{вх}(f)$", fontsize=10)
        ax.set_xlabel(r"Частота f, Гц", fontsize=10)
        ax.set_ylabel("$U_{вх}$, В", fontsize=10)

    elif ng == 3:
        ax.plot(fft_freq[0:num//2], sp_out[0:num//2], lw=1, color='#539caf', alpha=1)
        ax.set_xlim(0, max(fh, fc) * 1.05)
        # ax.set_ylim(min(sp_out[0:num//2]) * 1.1, max(sp_out[0:num//2]) * 1.05)
        if title: ax.set_title("(г) - $U_{вых}(f)$", fontsize=10)
        ax.set_xlabel(r"Частота f, Гц", fontsize=10)
        ax.set_ylabel("$U_{вых}$, В", fontsize=10)

    elif ng == 4:
        if type_signal == 5:
            pass
        else:
            color = 'tab:blue'
            ax.plot(freq, Gama.imag, lw=1, color=color, alpha=1)
            ax.set_xlim(0, max(fh, fc) * 1.1)
            if title: ax.set_title(r"(д) - $Z_0(f)$, $\varphi(f)$", fontsize=10)
            ax.set_xlabel(r"Частота f, Гц", fontsize=10)
            ax.set_ylabel(r"$\varphi$, рад.", color=color, fontsize=10)
            ax.tick_params(axis='y', labelcolor=color)

            ax = ax.twinx()
            color = 'tab:red'
            ax.plot(freq, abs(Zw), lw=1, color=color, alpha=1)
            ax.set_ylim(-0.1 * Z0, min(max(Zw.real), 3.5 * Z0))
            ax.set_ylabel("$Z_0$, Ом", color=color, fontsize=10)
            ax.vlines(f0, -0.1 * Z0, Z0, color='tab:olive', linestyles='dashdot', lw=1)
            ax.hlines(Z0, f0, freq[-1], color='tab:olive', linestyles='dashdot', lw=1)
            ax.tick_params(axis='y', labelcolor=color)

    elif ng == 5:
        ax.set_prop_cycle(color=['r', 'g', 'b', 'y', 'c'], linestyle=['-', '--', ':', '-.', '-'], lw=5 * [0.5])
        ax.set_alpha(0.5)
        cells = np.linspace(0, Nc, Nc + 1)
        z_spl = np.linspace(0, Nc, (Nc + 1) * 10)
        for i in range(npp):
            spl = make_interp_spline(cells, Vs[i], k=3)
            ax.plot(z_spl, spl(z_spl))
        ax.set_xlim(0, Nc)
        if title: ax.set_title("(е) - $U_i$", fontsize=10)
        ax.set_xlabel("Номер ячейки $(i)$", fontsize=10)
        ax.set_ylabel("$U_i$, В", fontsize=10)

drawAllGraph()
