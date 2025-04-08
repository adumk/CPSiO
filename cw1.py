import matplotlib
<<<<<<< HEAD
from numpy import ndarray
=======
>>>>>>> 155b980 (Zadanie 4)
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
def load_signal(data: np.ndarray, lead_index: int, use_time_column=False) -> (ndarray, ndarray):
    """
    :param data: 2D/1D array
    :param lead_index: Column to use as values
    :param use_time_column: Use first column as time entry
    :return: Pair of arrays (time, value)
    """
    if data.ndim == 1:
        return np.arange(len(data)), data

    if use_time_column:
        if lead_index == 0:
            raise Exception("Proba wczytania columny czasu jako wartosci")
        time_column = data[:,0]
    else:
        time_column = np.arange(data.shape[0])

    return time_column, data[:, lead_index]

def plot_ekg(time: ndarray, signals: list[list], start_time: float = None, end_time: float = None, min_value: float = None, max_value: float = None) -> None:
    """
    :param time: 1D array of keys
    :param signals: Array of arrays with values
    :param start_time: Start key to cut values
    :param end_time: End key to cut values
    :param min_value: Max value of signal to show
    :param max_value: Min value of singal to show
    """
    signals_to_show = []
    time_cut = cut_signal(time,signals[0],start_time, end_time)[0]
    for sig in signals:
        signals_to_show.append(cut_signal(time,sig,start_time, end_time, min_value, max_value)[1])

    show_figure(time_cut, signals_to_show)

def plot_signals(time_col: list, signals: list[list]) -> None:
    """
    :param time_col: Array of keys
    :param signals: Array of arrays with values
    """
    for idx, sig in enumerate(signals):
        plt.plot(time_col, sig, label=f'Sygnał EKG {idx+1}')

def cut_signal(time: ndarray, signal: list, start_time: float = None, end_time: float = None, min_value: float = None, max_value: float = None) -> (ndarray, ndarray):
    """
    :param time: Array of keys
    :param signal: Array of values
    :param start_time: Start key to cut values
    :param end_time: End key to cut values
    :param min_value: Max value of signal to show
    :param max_value: Min value of singal to show
    :return:
    """
    mask = None
    if start_time is not None and end_time is not None:
        mask = (time <= end_time) & (time >= start_time)
    elif end_time is not None:
        mask = time <= end_time
    elif start_time is not None:
        mask = time >= start_time

    if min_value is not None or max_value is not None:
        for idx,x in enumerate(signal):
            if min_value is not None:
                if x < min_value:
                    signal[idx] = min_value
            if max_value is not None:
                if x > max_value:
                    signal[idx] = max_value

    if mask is not None:
        return time[mask], signal[mask]

    return time, signal

def show_figure(keys: list, values, x_label: str = 'Czas [s]', y_label: str = 'Amplituda',
                title: str = 'Wizualizacja sygnału EKG') -> None:
    """
    :param keys: Array of keys
    :param values: Array of arrays with values
    :param x_label: X label
    :param y_label: T label
    :param title: Title of figure
    """
    plt.figure(figsize=(10, 5))
    plot_signals(keys, values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
=======

def load_ekg_1(filename):
    data = np.loadtxt(filename)
    if data.shape[1] == 2:  # Jeśli plik zawiera kolumnę czasu
        time, signal = data[:, 0], data[:, 1]
    else:
        time = np.arange(len(data))  # Jeśli brak kolumny czasu, generujemy indeksy próbek
        signal = data[:, 0]  # Pierwsza kolumna jako sygnał
    return time, signal

def load_ekg_2(filename, lead_index=1):
    """
    Wczytuje dane EKG z pliku.

    :param filename: Nazwa pliku z danymi EKG.
    :param lead_index: Indeks kolumny zawierającej sygnał EKG (domyślnie 1).
    :return: time (oś czasu), signal (wybrane odprowadzenie EKG)
    """
    data = np.loadtxt(filename)

    # Jeśli plik ma tylko jedną kolumnę (sam sygnał)
    if data.ndim == 1:
        time = np.arange(len(data))
        signal = data
    else:
        # Jeśli plik ma wiele kolumn, domyślnie używamy podanej kolumny (lead_index)
        time = np.arange(data.shape[0])  # Indeksy jako czas
        signal = data[:, lead_index]  # Pobranie wybranego odprowadzenia

    return time, signal


def plot_ekg(time, signal, start_time=None, end_time=None):
    if start_time is not None and end_time is not None:
        mask = (time >= start_time) & (time <= end_time)
        time, signal = time[mask], signal[mask]

    plt.figure(figsize=(10, 5))
    plt.plot(time, signal, label='Sygnał EKG')
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.title("Wizualizacja sygnału EKG")
>>>>>>> 155b980 (Zadanie 4)
    plt.legend()
    plt.grid()
    plt.show()

<<<<<<< HEAD
def save_ekg(filename: str, time: ndarray, signal: list, start_time: float, end_time: float) -> None:
    """
    :param filename: Destination file name
    :param time: Array of keys
    :param signal: Array of values
    :param start_time: Start key to cut values
    :param end_time: End key to cut values
    """
    time, signal = cut_signal(time, signal, start_time, end_time)
    segment = np.column_stack((time, signal))
    np.savetxt(filename, segment, fmt='%f', delimiter=' ', header='Czas Amplituda')
    print(f"Zapisano wycinek sygnału do pliku: {filename}")

def generate_sinus(samples_count: int=65536, fs: int=1000, f:int=50) -> (ndarray, ndarray):
    sample_t = 1 / fs  # Okres próbkowania
    t = np.arange(0, samples_count) * sample_t  # Wektor czasu (t)
    x = np.sin(2 * np.pi * f * t)  # Sygnał sinusoidalny
    return t, x

def get_signal_spectrum_fft(signal: list, fs: int, samples_count: int) -> (ndarray, ndarray):
    # Obliczenie FFT sygnału
    signal_fft = np.fft.fft(signal)

    # Obliczenie częstotliwości
    frequencies = np.fft.fftfreq(samples_count, 1 / fs)

    # Wyznaczenie widma amplitudowego (tylko dodatnie częstotliwości)
    amplitude_spectrum = np.abs(signal_fft[:samples_count // 2])
    frequencies_positive = frequencies[:samples_count // 2]

    return frequencies_positive, amplitude_spectrum

def get_full_signal_spectrum_fft(signal: list, fs: int, samples_count: int) -> (ndarray, ndarray):
    signal_fft = np.fft.fft(signal)  # Pobieramy pełne FFT
    frequencies = np.fft.fftfreq(samples_count, 1 / fs)  # Pełna siatka częstotliwości
    return frequencies, signal_fft  # Zwracamy PEŁNE wartości FFT

def get_signal_from_spectrum_ifft(signal_spectrum: list) -> ndarray:
    return np.real(np.fft.ifft( np.concatenate((signal_spectrum, signal_spectrum[::-1]))))

def cwiczenie1() -> None:
    _filenames = ["ekg100.txt", "ekg1.txt", "ekg_noise.txt"]
    time = []
    signal = []
    for _filename in _filenames:
        data = np.loadtxt(_filename)

        if _filename == "ekg100.txt":
            time, signal = load_signal(data, 3)
            plot_ekg(time, [signal], start_time=0.0, end_time=1000, min_value=-0.4, max_value=0.4)
        elif _filename == "ekg1.txt":
            signals = []
            for i in range(12):
                time, sig = load_signal(data, i)
                signals.append(sig)
            plot_ekg(time, signals, start_time=0.0, end_time=500)
        if _filename == "ekg_noise.txt":
            time, signal = load_signal(data, 1, True)
            plot_ekg(time, [signal], start_time=0.0, end_time=2)

    # Zapisanie wycinka sygnału do pliku
    save_ekg("ekg_segment.txt", time, signal, start_time=0.0, end_time=10)

def cwiczenie2_a() -> None:
    # Parametry sygnału
    samples_count = 65536 # Ilosc probek
    fs = 1000  # Częstotliwość próbkowania (Hz)
    f = 50  # Częstotliwość fali sinusoidalnej (Hz)
    time, signal = generate_sinus(samples_count, fs, f)
    time, signal_cut = cut_signal(time, signal, 0.1, 0.25)
    show_figure(time,[signal_cut],'Czas [s]','Amplituda','Wycinek sygnału sinusoidalnego o częstotliwości 50 Hz')

def cwiczenie2_b() -> None:
    # Parametry sygnału
    samples_count = 65536  # Ilosc probek
    fs = 1000  # Częstotliwość próbkowania (Hz)
    f = 50  # Częstotliwość fali sinusoidalnej (Hz)
    time, signal = generate_sinus(samples_count, fs, f)
    time, signal_cut = cut_signal(time, signal, 0.1, 0.25)
    freq, values_50Hz = get_signal_spectrum_fft(signal, fs, samples_count)
    show_figure(freq,[values_50Hz],'Czestotliwość [Hz]','Amplituda','Widmo amplitudowe sygnału 50 Hz')

def cwiczenie2_c(fs: int) -> None:
    samples_count = 65536  # Ilosc probek
    time, signal_50Hz = generate_sinus(samples_count, fs, 50)
    time, signal_60Hz = generate_sinus(samples_count, fs, 60)
    singal_mix = signal_50Hz + signal_60Hz

    time_cut, signal = cut_signal(time, singal_mix, 0.1, 0.25)
    show_figure(time_cut,[signal],'Czas [s]','Amplituda',f'Wycinek sygnalu 50Hz+60Hz, czestotliwosc probkowania {fs} Hz')

    samples_count = 65536  # Ilosc probek
    time, signal_50Hz = generate_sinus(samples_count, fs, 50)
    time, signal_60Hz = generate_sinus(samples_count, fs, 60)
    singal_mix = signal_50Hz + signal_60Hz
    freq, values = get_signal_spectrum_fft(singal_mix, fs, samples_count)
    show_figure(freq,[values],'Czestotliwość [Hz]','Amplituda',f'Widmo amplitudowe sygnału 50Hz+60Hz, czestotliwosc probkowania {fs} Hz')

def cwiczenie2_d():
    cwiczenie2_c(2000)
    cwiczenie2_c(1500)
    cwiczenie2_c(500)
    cwiczenie2_c(200)
    cwiczenie2_c(130)
    cwiczenie2_c(90)
    cwiczenie2_c(20)

def cwiczenie2_e(fs : int):
    # samples_count = 65536  # Ilosc probek
    # fs = 100000  # Częstotliwość próbkowania (Hz)
    # f = 50  # Częstotliwość fali sinusoidalnej (Hz)
    # time, signal = generate_sinus(samples_count, fs, f)
    # freq, values_50Hz = get_signal_spectrum_fft(signal, fs, samples_count)
    # signal = get_signal_from_spectrum_ifft(values_50Hz)
    # show_figure(time,[signal],'Czas [s]','Amplituda','Odzyskany sygnal z widma')

    samples_count = 65536  # Ilosc probek
    time, signal_50Hz = generate_sinus(samples_count, fs, 50)

    freq, values = get_full_signal_spectrum_fft(signal_50Hz, fs, samples_count)

    reconstructed_signal = np.fft.ifft(values).real

    time = np.arange(samples_count) / fs


    plt.figure(figsize=(10, 5))
    plt.plot(time, signal_50Hz, label="Oryginalny sygnał")
    plt.plot(time, reconstructed_signal, '--', label="Zrekonstruowany sygnał (IFFT)", alpha=0.7)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.title("Porównanie sygnału oryginalnego i po IFFT, fala sinusoidalna 50Hz")
    plt.show()

    samples_count = 65536  # Ilosc probek
    time, signal_50Hz = generate_sinus(samples_count, fs, 50)
    time, signal_60Hz = generate_sinus(samples_count, fs, 60)
    singal_mix = signal_50Hz + signal_60Hz

    freq, values = get_full_signal_spectrum_fft(singal_mix, fs, samples_count)

    reconstructed_signal = np.fft.ifft(values).real

    time = np.arange(samples_count) / fs


    plt.figure(figsize=(10, 5))
    plt.plot(time, singal_mix, label="Oryginalny sygnał")
    plt.plot(time, reconstructed_signal, '--', label="Zrekonstruowany sygnał (IFFT)", alpha=0.7)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.legend()
    plt.title("Porównanie sygnału oryginalnego i po IFFT, mieszanina fal sinusoidalnych 50Hz i 60Hz")
    plt.show()


def cwiczenie2():
    cwiczenie2_a()
    cwiczenie2_b()
    cwiczenie2_c(1000)
    cwiczenie2_d()
    cwiczenie2_e(100000)

def cwiczenie3():
    data = np.loadtxt("ekg100.txt")
    time, signal = load_signal(data, 0)
    show_figure(time,[signal])

    for fs in [100000, 40000, 30000, 1000]:
        # Obliczenie widma
        freq, values = get_signal_spectrum_fft(signal, fs, len(signal))
        show_figure(freq, [values], 'Częstotliwość [Hz]', 'Amplituda', f'Widmo amplitudowe sygnału, częstotliwość próbkowania {fs} Hz')

        # Pełne widmo + IFFT
        freq, values = get_full_signal_spectrum_fft(signal, fs, len(signal))
        reconstructed_signal = np.fft.ifft(values).real

        # Różnica między oryginalnym a zrekonstruowanym sygnałem
        difference = signal - reconstructed_signal

        # Poprawny wektor czasu
        time = np.arange(len(signal)) / fs

        # Wykres sygnału oryginalnego i zrekonstruowanego
        plt.figure(figsize=(10, 5))
        plt.plot(time, signal, label="Oryginalny sygnał")
        plt.plot(time, reconstructed_signal, '--', label="Zrekonstruowany sygnał (IFFT)", alpha=0.7)
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda")
        plt.legend()
        plt.title(f"Porównanie sygnału oryginalnego i po IFFT (fs={fs} Hz)")
        plt.show()

        max_error = np.max(np.abs(signal - reconstructed_signal))
        print("Maksymalny błąd rekonstrukcji:", max_error)

        mean_error = np.mean(np.abs(signal - reconstructed_signal))
        print("Średni błąd rekonstrukcji:", mean_error)

        print("Zakres wartości różnicy:", np.min(signal - reconstructed_signal),
              np.max(signal - reconstructed_signal))
        print("\n")

        # Wykres różnicy sygnałów
        plt.figure(figsize=(10, 5))
        plt.plot(time, difference, label="Różnica sygnałów", color='red')
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda różnicy")
        plt.legend()
        plt.title(f"Różnica sygnału oryginalnego i IFFT (fs={fs} Hz)")
        plt.ylim(-1e-14, 1e-14)
        plt.show()

def cwiczenie4():
    data = np.loadtxt("ekg_noise.txt")
    time, signal = load_signal(data,1 )
    show_figure(time,[signal])

# TODO: Kontynuować zadania (cwiczenie 4)
#cwiczenie1()
cwiczenie2()
cwiczenie3()
#cwiczenie4()
=======

def save_ekg(filename, time, signal, start_time, end_time):
    mask = (time >= start_time) & (time <= end_time)
    segment = np.column_stack((time[mask], signal[mask]))
    np.savetxt(filename, segment, fmt='%f', delimiter=' ', header='Czas Amplituda')
    print(f"Zapisano wycinek sygnału do pliku: {filename}")


# Wczytanie danych
filename = "ekg100.txt"
if filename == "ekg1.txt":
    time, signal = load_ekg_2(filename, lead_index=3)
    plot_ekg(time, signal, start_time=0.0, end_time=1000)
elif filename == "ekg100.txt":
    time, signal = load_ekg_2(filename)
    plot_ekg(time, signal, start_time=0.0, end_time=500)
else:
    time, signal = load_ekg_1(filename)
    plot_ekg(time, signal, start_time=0.0, end_time=2)

# Przykładowa wizualizacja wycinka (np. od 0.5s do 1.5s)


# Zapisanie wycinka sygnału do pliku
save_ekg("ekg_segment.txt", time, signal, start_time=0.0, end_time=10)
>>>>>>> 155b980 (Zadanie 4)
