import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt


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
    plt.legend()
    plt.grid()
    plt.show()


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
