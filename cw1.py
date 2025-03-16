import matplotlib
from numpy import ndarray
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

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

def show_figure(time: list, signals: list, x_label: str = 'Czas [s]', y_label: str = 'Amplituda',
                title: str = 'Wizualizacja sygnału EKG') -> None:
    """
    :param time: Array of keys
    :param signals: Array of arrays with values
    :param x_label: X label
    :param y_label: T label
    :param title: Title of figure
    """
    plt.figure(figsize=(10, 5))
    plot_signals(time,signals)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

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


# Wczytanie danych

def cwiczenie1():
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

def cwiczenie2():
    # Parametry sygnału
    fs = 1000  # Częstotliwość próbkowania (Hz)
    f = 50  # Częstotliwość fali sinusoidalnej (Hz)
    T = 1 / fs  # Okres próbkowania
    t = np.arange(0, 65536) * T  # Wektor czasu (t)
    x = np.sin(2 * np.pi * f * t)  # Sygnał sinusoidalny

    t, x = cut_signal(t, x,0.5, 0.55)
    show_figure(t,[x],'Czas [s]','Amplituda','Sygnał sinusoidalny o częstotliwości 50 Hz')


cwiczenie1()
cwiczenie2()