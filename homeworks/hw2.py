import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def f2w(f):
    return 2.0 * math.pi * f

def Z1(f, C1):
    return 2.0 / (1j * f2w(f) * C1)

def Z2(f, C2):
    return 1.0 / (1j * f2w(f) * C2)

def Z3(f, L):
    return 1.0j * f2w(f) * L

def Gam(f, L, C1, C2):
    ZY = (Z2(f, C2) + Z3(f, L)) / Z1(f, C1)
    return 2.0 * np.arcsinh(np.sqrt(ZY))

def Zw(f, L, C1, C2):
    num = Z1(f, C1)**2 * (Z2(f, C2) + Z3(f, L))
    den = 2 * Z1(f, C1) + Z2(f, C2) + Z3(f, L)
    return np.sqrt(num / den)

def bandpass_filter(time, signal, fl, fh):
    n = len(signal)
    dt = time[1] - time[0]
    freqs = np.fft.fftfreq(n, dt)
    spectrum = np.fft.fft(signal)

    for i in range(n):
        if not (fl < abs(freqs[i]) < fh):
            spectrum[i] = 0j

    return np.fft.ifft(spectrum).real


fs = 10      # Базовая частота сигнала
T = 0.2      # Длительность сигнала
n = 10000    # Число точек дискретизации
mod_depth = 0.5  # Глубина модуляции
Z0 = 10      # Характеристическое сопротивление линии
Nc = 100     # Число ячеек в линии

# Инициализация массивов
time = np.linspace(0, T, n)
freqs = np.fft.fftfreq(n, T/n)

sig1 = np.zeros(n)  # Аналоговый (речевой) сигнал
sig2 = np.zeros(n)  # Цифровой сигнал
sam1 = np.zeros(n)  # Несущая 10*fs
sam2 = np.zeros(n)  # Несущая 50*fs

# Генерация сигналов
for i, t in enumerate(time):
    sig1[i] = 5.0 * math.sin(1*f2w(fs)*t) + 7.0 * math.sin(4*f2w(fs)*t) + 9.0 * math.sin(7*f2w(fs)*t)

    sig2[i] = 6.5 * math.sin(2*f2w(fs)*t) + 8.5 * math.sin(3*f2w(fs)*t) + 10.0 * math.sin(5*f2w(fs)*t)

    sam1[i] = math.sin(f2w(10*fs)*t)
    sam2[i] = math.sin(f2w(50*fs)*t)


# Дискретизация по времени
d_t = 1 / (2 * 10 * fs)
d_n = int(T / d_t)
d_sig = np.zeros(d_n)
d_time = np.zeros(d_n)

for i in range(d_n):
    idx = i * int(n/d_n)
    d_sig[i] = sig2[min(idx, n-1)]
    d_time[i] = i * d_t

# Квантование по уровню (2 уровня)
min_level = min(sig2)
max_level = max(sig2)
d_U = (max_level - min_level) / 1  # Для 2 уровней

q_sig = np.zeros(d_n)
for i in range(d_n):
    value = d_sig[i]
    if value >= (min_level + max_level)/2:
        q_sig[i] = max_level
    else:
        q_sig[i] = min_level

# Формирование цифрового сигнала
lengthOfBits = int(n / d_n)
sig2d = np.zeros(n)

for i in range(d_n):
    start_idx = i * lengthOfBits
    end_idx = (i+1) * lengthOfBits
    if end_idx > n: end_idx = n
    sig2d[start_idx:end_idx] = q_sig[i]

# Визуализация исходных сигналов
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.title('Речевой сигнал')
plt.plot(time, sig1)

plt.subplot(2, 2, 2)
plt.title('Спектр речевого сигнала')
plt.xlim(0, 8*fs)
spectrum = np.fft.fft(sig1)
plt.plot(freqs, np.abs(spectrum)/n*2)

plt.subplot(2, 2, 3)
plt.title('Цифровой сигнал')
plt.plot(time, sig2, '--', color='gray')
plt.scatter(d_time, q_sig, color='red')
plt.plot(time, sig2d)

plt.subplot(2, 2, 4)
plt.title('Спектр цифрового сигнала')
plt.xlim(0, 100*fs)
spectrum = np.fft.fft(sig2d)
plt.plot(freqs, np.abs(spectrum)/n*2)

plt.tight_layout()
plt.show()


interp_func = interp1d(d_time, q_sig, kind='cubic', fill_value="extrapolate")
sig2a = interp_func(time)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title("ЦАП - восстановленный сигнал")
plt.plot(time, sig2a)

plt.subplot(1, 2, 2)
plt.title('Спектр после ЦАП')
plt.xlim(fs, 12*fs)
spectrum = np.fft.fft(sig2a)
plt.plot(freqs, np.abs(spectrum)/n*2)

plt.tight_layout()
plt.show()


# Амплитудная модуляция
for i in range(n):
    sam1[i] = sam1[i] * (1 + mod_depth * sig1[i] / 2.0)
    sam2[i] = sam2[i] * (1 + mod_depth * sig2a[i] / 2.0)

# Формирование группового сигнала
grp = bandpass_filter(time, sam1, 3*fs-5, 17*fs+5) + bandpass_filter(time, sam2, (50-13)*fs-5, (50+13)*fs+5)

# Визуализация модулированных сигналов
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title("Амплитудная модуляция")
plt.plot(time, sam1, label='10fs')
plt.plot(time, sam2, label='50fs')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Спектр АМ сигналов')
plt.xlim(fs, 70*fs)
spectrum1 = np.fft.fft(sam1)
spectrum2 = np.fft.fft(sam2)
plt.plot(freqs, np.abs(spectrum1)/n*2)
plt.plot(freqs, np.abs(spectrum2)/n*2)

plt.tight_layout()
plt.show()

# Визуализация группового сигнала
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title("Групповой сигнал")
plt.plot(time, grp)

plt.subplot(1, 2, 2)
plt.title('Спектр группового сигнала')
plt.xlim(fs, 70*fs)
spectrum = np.fft.fft(grp)
plt.plot(freqs, np.abs(spectrum)/n*2)

plt.tight_layout()
plt.show()


# Расчет параметров линии
fl = 0.5 * fs
fh = 70.5 * fs
fc = (fl + fh) / 2
wl = f2w(fl)
wc = f2w(fc)
wh = f2w(fh)

L = math.sqrt(Z0**2 * wc**2 * (2*wh**2 - wl**2 - wc**2) /
             ((wh**2 - wl**2)**2 * (wc**2 - wl**2)))
C1 = 2.0 / (L * (wh**2 - wl**2))
C2 = 1.0 / (wl**2 * L)
G = 0

print(f'Параметры ячейки ЛП:\nC1 = {C1:.6f}\nC2 = {C2:.6f}\nL = {L:.6f}')

# Расчет характеристик линии
freq_lp = np.linspace(0.8*fl, fh*1.2, int(T*fh))
Gama_val = Gam(freq_lp, L, C1, C2)
Zw_val = Zw(freq_lp, L, C1, C2)
dF = (Gam(freq_lp+0.1, L, C1, C2).imag - Gam(freq_lp-0.1, L, C1, C2).imag)/0.2

# Визуализация характеристик линии
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.title("Постоянная распространения")
plt.plot(freq_lp, Gama_val.real, 'b', label=r'$\alpha(f)$')
plt.plot(freq_lp, Gama_val.imag, 'r', label=r'$\phi(f)$')
plt.legend()

plt.subplot(2, 2, 2)
plt.title("Характеристическое сопротивление")
plt.plot(freq_lp, np.abs(Zw_val), label='|Z|')
plt.plot(freq_lp, Zw_val.real, '--', label='Re')
plt.plot(freq_lp, Zw_val.imag, '--', label='Im')
plt.legend()

plt.subplot(2, 2, 3)
plt.title("Производная фазовой постоянной")
plt.plot(freq_lp, dF)

plt.tight_layout()
plt.show()

# Моделирование линии передачи
dt = T / n
aU = np.zeros(Nc)      # Напряжения на C2
dU = np.zeros(Nc)      # Производные напряжений на C2
aV = np.zeros(Nc+1)    # Напряжения на C1
dV = np.zeros(Nc+1)    # Производные напряжений на C1

lp_in = np.zeros(n)    # Входные напряжения
lp_out = np.zeros(n)   # Выходные напряжения

dpp = 20  # Число итераций на шаг

def d_signal(t):
    idx = min(n-1, int(t/dt))
    return (grp[idx] - grp[idx-1])/dt if idx > 0 else 0

for it in range(n):
    t = dt * it
    for _ in range(dpp):
        # Граничные условия
        dV[0] += (1.0/(L*C1)*(aV[1]-aV[0]+aU[0]) +
                 1.0/(Z0*C1)*(d_signal(t) - dV[0])) * dt/dpp

        # Уравнения для внутренних ячеек
        for ic in range(Nc):
            dU[ic] += (1.0/(L*C2)*(aV[ic]-aV[ic+1]-aU[ic]) -
                      G/C2*dU[ic]) * dt/dpp
            if ic > 0:
                dV[ic] += (0.5/(L*C1)*(aV[ic-1]-2*aV[ic]+aV[ic+1]+aU[ic]-aU[ic-1])) * dt/dpp

        dV[Nc] += (1.0/(L*C1)*(aV[Nc-1]-aV[Nc]-aU[Nc-1]) +
                  1.0/(Z0*C1)*(-dV[Nc])) * dt/dpp

        # Обновление напряжений
        for ic in range(Nc):
            aV[ic] += dV[ic] * dt/dpp
            aU[ic] += dU[ic] * dt/dpp
        aV[Nc] += dV[Nc] * dt/dpp

    lp_in[it] = aV[0]
    lp_out[it] = aV[Nc]

# Визуализация результатов прохождения через линию
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title("Входной и выходной сигналы на ЛП")
plt.plot(time, lp_in, label='Вход')
plt.plot(time, lp_out, label='Выход')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Спектр после ЛП')
plt.xlim(0, 70*fs)
spectrum = np.fft.fft(lp_out)
plt.plot(freqs, np.abs(spectrum)/n*2)

plt.tight_layout()
plt.show()


# Выделение канальных сигналов
rch1 = bandpass_filter(time, lp_out, 3*fs-5, 17*fs+5)
rch2 = bandpass_filter(time, lp_out, (50-13)*fs-5, (50+13)*fs+5)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title("Выделение канальных сигналов")
plt.plot(time, rch1, label='Канал 1')
plt.plot(time, rch2, label='Канал 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Спектры канальных сигналов')
plt.xlim(0, 70*fs)
spectrum1 = np.fft.fft(rch1)
spectrum2 = np.fft.fft(rch2)
plt.plot(freqs, np.abs(spectrum1)/n*2)
plt.plot(freqs, np.abs(spectrum2)/n*2)

plt.tight_layout()
plt.show()

# Демодуляция
mch1 = np.zeros(n)
mch2 = np.zeros(n)

for i in range(n):
    mch1[i] = rch1[i] * math.cos(f2w(10*fs)*time[i])
    mch2[i] = rch2[i] * math.cos(f2w(50*fs)*time[i])

norm_factor1 = 2.0 / mod_depth
norm_factor2 = 2.0 / mod_depth
mch1 *= norm_factor1
mch2 *= norm_factor2

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title('Амплитудная демодуляция')
plt.plot(time, mch1, label='Канал 1')
plt.plot(time, mch2, label='Канал 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Спектры демодуляции')
plt.xlim(0, 15*fs)
spectrum1 = np.fft.fft(mch1)
spectrum2 = np.fft.fft(mch2)
plt.plot(freqs, np.abs(spectrum1)/n*2)
plt.plot(freqs, np.abs(spectrum2)/n*2)

plt.tight_layout()
plt.show()

# Восстановление сигналов
rsig1 = bandpass_filter(time, mch1, 1, 7*fs+1)
rsig2 = bandpass_filter(time, mch2, 1, 15*fs+1)

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title('Восстановленные сигналы')
plt.plot(time, rsig1, label='Аналоговый')
plt.plot(time, rsig2, label='Цифровой')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Спектры после ФНЧ')
plt.xlim(0, 13*fs)
spectrum1 = np.fft.fft(rsig1)
spectrum2 = np.fft.fft(rsig2)
plt.plot(freqs, np.abs(spectrum1)/n*2)
plt.plot(freqs, np.abs(spectrum2)/n*2)

plt.tight_layout()
plt.show()

# АЦП для восстановленного цифрового сигнала
d_sig_restored = np.zeros(d_n)
for i in range(d_n):
    idx = i * int(n/d_n)
    d_sig_restored[i] = rsig2[min(idx, n-1)]

q_sig_restored = np.zeros(d_n)
for i in range(d_n):
    value = d_sig_restored[i]
    if value >= (min_level + max_level)/2:
        q_sig_restored[i] = max_level
    else:
        q_sig_restored[i] = min_level

rsig2d = np.zeros(n)
for i in range(d_n):
    start_idx = i * lengthOfBits
    end_idx = (i+1) * lengthOfBits
    if end_idx > n: end_idx = n
    rsig2d[start_idx:end_idx] = q_sig_restored[i]

# Финальная визуализация
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title('Восстановленный цифровой сигнал')
plt.plot(time, rsig2, '--', color='gray')
plt.scatter(d_time, q_sig_restored, color='red')
plt.plot(time, rsig2d)

plt.subplot(1, 2, 2)
plt.title('Спектр восстановленного сигнала')
plt.xlim(0, 100*fs)
spectrum = np.fft.fft(rsig2d)
plt.plot(freqs, np.abs(spectrum)/n*2)

plt.tight_layout()
plt.show()
