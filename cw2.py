import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import scipy.ndimage as ndi


def wczytaj_obraz(nazwa_pliku):
    """Wczytuje obraz z pliku o podanej nazwie."""
    return Image.open(nazwa_pliku)


def wyswietl_obraz(obraz):
    """Wyświetla obraz."""
    plt.imshow(obraz, cmap='gray')
    plt.axis('off')  # Ukrywa osie
    plt.show()


def wykres_poziomych_linii(obraz, wspolrzedna_y):
    """Rysuje wykres poziomu szarości wzdłuż poziomej linii o zadanej współrzędnej y."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)

    if wspolrzedna_y < 0 or wspolrzedna_y >= obraz_array.shape[0]:
        print("Nieprawidłowa współrzędna y.")
        return

    # Pobieramy wartości pikseli wzdłuż poziomej linii
    poziomy_odcinek = obraz_array[wspolrzedna_y, :]

    # Tworzymy wykres
    plt.plot(poziomy_odcinek)
    plt.title(f'Poziom szarości na linii y={wspolrzedna_y}')
    plt.xlabel('Pozycja na osi X')
    plt.ylabel('Poziom szarości')
    plt.show()


def wykres_pionowych_linii(obraz, wspolrzedna_x):
    """Rysuje wykres poziomu szarości wzdłuż pionowej linii o zadanej współrzędnej x."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)

    if wspolrzedna_x < 0 or wspolrzedna_x >= obraz_array.shape[1]:
        print("Nieprawidłowa współrzędna x.")
        return

    # Pobieramy wartości pikseli wzdłuż pionowej linii
    pionowy_odcinek = obraz_array[:, wspolrzedna_x]

    # Tworzymy wykres
    plt.plot(pionowy_odcinek)
    plt.title(f'Poziom szarości na linii x={wspolrzedna_x}')
    plt.xlabel('Pozycja na osi Y')
    plt.ylabel('Poziom szarości')
    plt.show()


def wybierz_podobraz(obraz, x_start, y_start, szerokosc, wysokosc):
    """Wybiera podobszar obrazu i zapisuje go do nowego pliku."""
    obraz_podobraz = obraz.crop((x_start, y_start, x_start + szerokosc, y_start + wysokosc))
    return obraz_podobraz


def zapis_obraz_do_pliku(obraz, nazwa_pliku):
    """Zapisuje obraz do pliku o zadanej nazwie."""
    obraz.save(nazwa_pliku)


def przeksztalcenie_mnozenie(obraz, c):
    """Mnożenie obrazu przez stałą T(r) = c * r."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)
    obraz_array = np.clip(c * obraz_array, 0, 255)  # Mnożymy przez stałą i ograniczamy do przedziału [0, 255]
    return Image.fromarray(obraz_array.astype(np.uint8))


def przeksztalcenie_logarytmiczne(obraz, c):
    """Transformacja logarytmiczna T(r) = c * log(1 + r)."""
    obraz_gray = obraz.convert('L')
    obraz_array = np.array(obraz_gray)
    obraz_array = c * np.log(1 + obraz_array)
    obraz_array = np.clip(obraz_array, 0, 255)  # Ograniczamy wartości do przedziału [0, 255]
    return Image.fromarray(obraz_array.astype(np.uint8))


def przeksztalcenie_kontrastowe(obraz, m, e):
    """Zmiana dynamiki skali szarości T(r) = 1 / (1 + (m / r) ^ e)."""
    obraz_gray = obraz.convert('L')
    obraz_array = np.array(obraz_gray)
    obraz_array = 1 / (1 + (m / (obraz_array + 1)) ** e)  # Dodajemy 1, by uniknąć dzielenia przez zero
    obraz_array = np.clip(obraz_array * 255, 0, 255)  # Skalowanie do zakresu [0, 255]
    return Image.fromarray(obraz_array.astype(np.uint8))


def przeksztalcenie_gamma(obraz, c, gamma):
    """Korekcja gamma T(r) = c * r ^ γ."""
    obraz_gray = obraz.convert('L')
    obraz_array = np.array(obraz_gray)
    obraz_array = c * (obraz_array / 255) ** gamma  # Normalizujemy do [0, 1] przed potęgą
    obraz_array = np.clip(obraz_array * 255, 0, 255)  # Skalowanie do zakresu [0, 255]
    return Image.fromarray(obraz_array.astype(np.uint8))


def wykres_transformacji(c, przeksztalcenie, nazwa_transformacji):
    """Rysuje wykres transformacji punktowej T(r) dla r ∈ [0, 255]."""
    r = np.linspace(0, 255, 256)
    s = przeksztalcenie(r, c)
    plt.plot(r, s, label=nazwa_transformacji)
    plt.title(f'Przekształcenie punktowe: {nazwa_transformacji}')
    plt.xlabel('Poziom szarości wejściowy (r)')
    plt.ylabel('Poziom szarości wyjściowy (s)')
    plt.legend()
    plt.grid(True)
    plt.show()


def przeksztalcenie_mnozenie_function(r, c):
    """Pomocnicza funkcja do wykresu mnożenia przez stałą."""
    return np.clip(c * r, 0, 255)


def przeksztalcenie_logarytmiczne_function(r, c):
    """Pomocnicza funkcja do wykresu transformacji logarytmicznej."""
    return np.clip(c * np.log(1 + r), 0, 255)


def przeksztalcenie_kontrastowe_function(r, m, e):
    """Pomocnicza funkcja do wykresu zmiany kontrastu."""
    return np.clip(1 / (1 + (m / (r + 1)) ** e) * 255, 0, 255)


def przeksztalcenie_gamma_function(r, c, gamma):
    """Pomocnicza funkcja do wykresu korekcji gamma."""
    return np.clip(c * (r / 255) ** gamma * 255, 0, 255)


def wyrownaj_histogram(obraz):
    """Wyrównywanie histogramu obrazu."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)

    # Obliczanie histogramu i skumulowanej funkcji rozkładu
    hist, bins = np.histogram(obraz_array.flatten(), bins=256, range=(0, 256))

    cdf = hist.cumsum()  # CDF (Skumulowana funkcja rozkładu)
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalizowanie CDF

    # Mapowanie wartości pikseli na nowy zakres
    obraz_eq = np.interp(obraz_array.flatten(), bins[:-1], cdf_normalized)

    # Przekształcenie z powrotem do obrazu
    obraz_eq = obraz_eq.reshape(obraz_array.shape)

    return Image.fromarray(np.uint8(obraz_eq))


def rysuj_histogram(obraz, tytul):
    """Rysowanie histogramu obrazu."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)

    plt.hist(obraz_array.flatten(), bins=256, range=(0, 256), color='black', alpha=0.7)
    plt.title(tytul)
    plt.xlabel('Poziom szarości')
    plt.ylabel('Liczba pikseli')
    plt.grid(True)


def lokalne_wyrownanie_histogramu(obraz, rozmiar_okna=15):
    """Lokalne wyrównywanie histogramu na obrazie."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)

    # Rozmiar okna dla lokalnego wyrównywania
    k, l = rozmiar_okna, rozmiar_okna
    wyniki = np.zeros_like(obraz_array)

    # Przechodzimy po obrazie w oknach
    for i in range(rozmiar_okna // 2, obraz_array.shape[0] - rozmiar_okna // 2):
        for j in range(rozmiar_okna // 2, obraz_array.shape[1] - rozmiar_okna // 2):
            okno = obraz_array[i - k // 2:i + k // 2 + 1, j - l // 2:j + l // 2 + 1]
            hist, bins = np.histogram(okno.flatten(), bins=256, range=(0, 256))
            cdf = hist.cumsum()  # Skumulowana funkcja rozkładu
            cdf_normalized = cdf * hist.max() / cdf.max()  # Normalizowanie CDF
            wyniki[i, j] = np.interp(obraz_array[i, j], bins[:-1], cdf_normalized)

    return Image.fromarray(np.uint8(wyniki))


def lokalne_poprawa_jakosci(obraz, rozmiar_okna=15):
    """Poprawa jakości obrazu oparta na lokalnych statystykach."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)

    k, l = rozmiar_okna, rozmiar_okna
    wyniki = np.zeros_like(obraz_array)

    # Przechodzimy po obrazie w oknach
    for i in range(rozmiar_okna // 2, obraz_array.shape[0] - rozmiar_okna // 2):
        for j in range(rozmiar_okna // 2, obraz_array.shape[1] - rozmiar_okna // 2):
            okno = obraz_array[i - k // 2:i + k // 2 + 1, j - l // 2:j + l // 2 + 1]
            # Obliczanie lokalnych statystyk: średnia i odchylenie standardowe
            lokalna_srednia = np.mean(okno)
            lokalne_odchylenie = np.std(okno)
            # Normalizacja: ustawienie piksela w nowym obrazie na podstawie lokalnej średniej
            wyniki[i, j] = np.clip(obraz_array[i, j] - lokalna_srednia + 128, 0, 255)

    return Image.fromarray(np.uint8(wyniki))


def dodaj_szum_sol_i_pieprz(obraz, poziom_szumu=0.05):
    """Dodaje szum typu 'sól i pieprz' do obrazu."""
    obraz_array = np.array(obraz)
    wysokosc, szerokosc = obraz_array.shape

    liczba_pikseli = int(wysokosc * szerokosc * poziom_szumu)
    for _ in range(liczba_pikseli):
        x = random.randint(0, wysokosc - 1)
        y = random.randint(0, szerokosc - 1)
        # Decyzja, czy to "sól" czy "pieprz"
        if random.random() < 0.5:
            obraz_array[x, y] = 0  # "Pieprz"
        else:
            obraz_array[x, y] = 255  # "Sól"

    return Image.fromarray(obraz_array)


def filtr_usredniajacy(obraz, maska_rozmiar=3):
    """Aplikacja liniowego filtra uśredniającego z maską o zadanym rozmiarze."""
    return obraz.filter(ImageFilter.BoxBlur(maska_rozmiar))


def filtr_medianowy(obraz, maska_rozmiar=3):
    """Aplikacja filtra medianowego."""
    return obraz.filter(ImageFilter.MedianFilter(maska_rozmiar))


def filtr_minimum(obraz, maska_rozmiar=3):
    """Aplikacja filtra minimum."""
    obraz_array = np.array(obraz)
    obraz_min = np.zeros_like(obraz_array)
    # Rozmiar okna
    k = maska_rozmiar // 2

    for i in range(k, obraz_array.shape[0] - k):
        for j in range(k, obraz_array.shape[1] - k):
            okno = obraz_array[i - k:i + k + 1, j - k:j + k + 1]
            obraz_min[i, j] = np.min(okno)

    return Image.fromarray(obraz_min)


def filtr_maksimum(obraz, maska_rozmiar=3):
    """Aplikacja filtra maksimum."""
    obraz_array = np.array(obraz)
    obraz_max = np.zeros_like(obraz_array)
    # Rozmiar okna
    k = maska_rozmiar // 2

    for i in range(k, obraz_array.shape[0] - k):
        for j in range(k, obraz_array.shape[1] - k):
            okno = obraz_array[i - k:i + k + 1, j - k:j + k + 1]
            obraz_max[i, j] = np.max(okno)

    return Image.fromarray(obraz_max)





def pokaz_obraz_i_histogram(obraz_przeksztalcony, tytul):
    """Funkcja pomocnicza do wyświetlania obrazu i histogramu."""
    plt.figure(figsize=(12, 6))

    # Wyświetlanie obrazu
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_przeksztalcony, cmap='gray')
    plt.title(f'{tytul} - Obraz')
    plt.axis('off')

    # Rysowanie histogramu
    plt.subplot(1, 2, 2)
    rysuj_histogram(obraz_przeksztalcony, f'{tytul} - Histogram')
    plt.show()


def zastosuj_filtr_usredniajacy(obraz, rozmiar_maski):
    """Zastosowanie filtra uśredniającego (BoxBlur) o zadanym rozmiarze maski."""
    return obraz.filter(ImageFilter.BoxBlur(rozmiar_maski))

def zastosuj_filtr_gaussowski(obraz, rozmiar_maski):
    """Zastosowanie filtra gaussowskiego o zadanym rozmiarze maski."""
    return obraz.filter(ImageFilter.GaussianBlur(rozmiar_maski))


######
def filtr_sobel(obraz):
    """Wykrywanie krawędzi z użyciem filtra Sobela."""
    # Sobel dla krawędzi poziomych (Gx) i pionowych (Gy)
    obraz_array = np.array(obraz.convert('L'))  # Konwersja obrazu do odcieni szarości

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Maska Sobela dla poziomych krawędzi
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Maska Sobela dla pionowych krawędzi

    # Konwolucja obrazu z maskami
    krawedzie_poziome = ndi.convolve(obraz_array, Gx)
    krawedzie_pionowe = ndi.convolve(obraz_array, Gy)

    # Łączenie wyników, aby uzyskać krawędzie ukośne
    krawedzie_ukosne = np.sqrt(krawedzie_poziome ** 2 + krawedzie_pionowe ** 2)

    return krawedzie_poziome, krawedzie_pionowe, krawedzie_ukosne


def filtr_laplasjan(obraz):
    """Zastosowanie filtra Laplasjanu do wyostrzania szczegółów."""
    obraz_array = np.array(obraz.convert('L'))
    laplasjan_maska = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Maska Laplasjanu

    # Konwolucja obrazu z maską Laplasjanu
    laplasjan_obraz = ndi.convolve(obraz_array, laplasjan_maska)

    # Wyostrzanie obrazu: oryginalny obraz + wyostrzona wersja (Laplasjan)
    wyostrzony_obraz = obraz_array + laplasjan_obraz
    return wyostrzony_obraz


def unsharp_masking(obraz, sigma=1.0, amount=1.5):
    """Zastosowanie unsharp masking do wyostrzania obrazu."""
    obraz_array = np.array(obraz.convert('L'))

    # Rozmywanie obrazu (sztuczne rozmycie)
    rozmyty_obraz = ndi.gaussian_filter(obraz_array, sigma)

    # Subtrakcja rozmytego obrazu od oryginalnego (wyostrzanie)
    sharp_image = obraz_array + amount * (obraz_array - rozmyty_obraz)
    return np.clip(sharp_image, 0, 255)


def high_boost_filtering(obraz, sigma=1.0, gain=1.5):
    """Zastosowanie high boost filtering do wyostrzania obrazu."""
    obraz_array = np.array(obraz.convert('L'))

    # Rozmywanie obrazu (sztuczne rozmycie)
    rozmyty_obraz = ndi.gaussian_filter(obraz_array, sigma)

    # High boost filtering (zwiększanie szczegółów)
    high_boost_image = obraz_array + gain * (obraz_array - rozmyty_obraz)
    return np.clip(high_boost_image, 0, 255)


def main():
    # 1. Wczytywanie obrazu
    nazwa_pliku = input("Podaj nazwę pliku obrazu: ")
    obraz = wczytaj_obraz(nazwa_pliku)

    # Wyświetlanie obrazu
    wyswietl_obraz(obraz)

    # 2. Sporządzanie wykresu poziomej linii
    wspolrzedna_y = int(input("Podaj współrzędną y (pozioma linia): "))
    wykres_poziomych_linii(obraz, wspolrzedna_y)

    # 3. Sporządzanie wykresu pionowej linii
    wspolrzedna_x = int(input("Podaj współrzędną x (pionowa linia): "))
    wykres_pionowych_linii(obraz, wspolrzedna_x)

    # 4. Wybór podobrazka
    x_start = int(input("Podaj współrzędną x początkową podobrazka: "))
    y_start = int(input("Podaj współrzędną y początkową podobrazka: "))
    szerokosc = int(input("Podaj szerokość podobrazka: "))
    wysokosc = int(input("Podaj wysokość podobrazka: "))

    podobraz = wybierz_podobraz(obraz, x_start, y_start, szerokosc, wysokosc)

    # Zapis podobrazka
    nazwa_zapisu = input("Podaj nazwę pliku do zapisania podobrazka: ")
    zapis_obraz_do_pliku(podobraz, nazwa_zapisu)

    print(f"Podobraz zapisany jako {nazwa_zapisu}.")

    # Wczytywanie obrazu
    nazwa_pliku = input("Podaj nazwę pliku obrazu: ")
    obraz = wczytaj_obraz(nazwa_pliku)

    # Przekształcenie
    # a) Mnożenie obrazu przez stałą
    c = float(input("Podaj wartość stałej c (przekształcenie mnożenia): "))
    obraz_mnozenie = przeksztalcenie_mnozenie(obraz, c)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_mnozenie, cmap='gray')
    plt.title(f'Mnożenie przez stałą c = {c}')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    wykres_transformacji(c, przeksztalcenie_mnozenie_function, "Mnożenie przez stałą")

    # Przekształcenie
    # b) Transformacja logarytmiczna
    c_log = float(input("Podaj wartość stałej c dla transformacji logarytmicznej: "))
    obraz_logarytmiczne = przeksztalcenie_logarytmiczne(obraz, c_log)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_logarytmiczne, cmap='gray')
    plt.title(f'Transformacja logarytmiczna c = {c_log}')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    wykres_transformacji(c_log, przeksztalcenie_logarytmiczne_function, "Transformacja logarytmiczna")

    # Przekształcenie
    # c) Zmiana kontrastu
    m = float(input("Podaj wartość parametru m (zmiana kontrastu): "))
    e = float(input("Podaj wartość parametru e (zmiana kontrastu): "))
    obraz_kontrastowe = przeksztalcenie_kontrastowe(obraz, m, e)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_kontrastowe, cmap='gray')
    plt.title(f'Zmiana kontrastu m = {m}, e = {e}')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    wykres_transformacji(m, lambda r, _: przeksztalcenie_kontrastowe_function(r, m, e), "Zmiana kontrastu")

    # Przekształcenie
    # d) Korekcja gamma
    c_gamma = float(input("Podaj wartość stałej c dla korekcji gamma: "))
    gamma = float(input("Podaj wartość parametru gamma dla korekcji gamma: "))
    obraz_gamma = przeksztalcenie_gamma(obraz, c_gamma, gamma)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_gamma, cmap='gray')
    plt.title(f'Korekcja gamma c = {c_gamma}, gamma = {gamma}')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    wykres_transformacji(c_gamma, lambda r, _: przeksztalcenie_gamma_function(r, c_gamma, gamma), "Korekcja gamma")
    plt.show()

    # Wczytywanie obrazów (np. obrazy zbyt ciemne i zbyt jasne)
    nazwa_pliku = input("Podaj nazwę pliku obrazu (ciemny lub jasny): ")
    obraz = wczytaj_obraz(nazwa_pliku)

    # Wyświetlanie obrazu przed wyrównaniem
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz, cmap='gray')
    plt.title('Obraz przed wyrównaniem')
    plt.axis('off')

    # Rysowanie histogramu przed wyrównaniem
    plt.subplot(1, 2, 2)
    rysuj_histogram(obraz, 'Histogram przed wyrównaniem')

    # Wyrównywanie histogramu
    obraz_wyr = wyrownaj_histogram(obraz)

    # Wyświetlanie obrazu po wyrównaniu
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_wyr, cmap='gray')
    plt.title('Obraz po wyrównaniu')
    plt.axis('off')

    # Rysowanie histogramu po wyrównaniu
    plt.subplot(1, 2, 2)
    rysuj_histogram(obraz_wyr, 'Histogram po wyrównaniu')

    # Wyświetlanie wyników
    plt.show()

    # Wczytywanie obrazu (np. obrazy zbyt ciemne i zbyt jasne)
    nazwa_pliku = input("Podaj nazwę pliku obrazu (ciemny lub jasny): ")
    obraz = wczytaj_obraz(nazwa_pliku)

    # Wyświetlanie obrazu przed przekształceniami
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz, cmap='gray')
    plt.title('Obraz przed przekształceniem')
    plt.axis('off')

    # Rysowanie histogramu przed przekształceniem
    plt.subplot(1, 2, 2)
    rysuj_histogram(obraz, 'Histogram przed przekształceniem')

    # Wykonanie lokalnego wyrównywania histogramu
    obraz_wyrownany = lokalne_wyrownanie_histogramu(obraz)

    # Wyświetlanie obrazu po lokalnym wyrównaniu histogramu
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_wyrownany, cmap='gray')
    plt.title('Obraz po lokalnym wyrównaniu histogramu')
    plt.axis('off')

    # Rysowanie histogramu po wyrównaniu
    plt.subplot(1, 2, 2)
    rysuj_histogram(obraz_wyrownany, 'Histogram po wyrównaniu')

    # Wykonanie poprawy jakości na podstawie lokalnych statystyk
    obraz_poprawiony = lokalne_poprawa_jakosci(obraz)

    # Wyświetlanie obrazu po poprawie jakości
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_poprawiony, cmap='gray')
    plt.title('Obraz po lokalnej poprawie jakości')
    plt.axis('off')

    # Rysowanie histogramu po poprawie jakości
    plt.subplot(1, 2, 2)
    rysuj_histogram(obraz_poprawiony, 'Histogram po poprawie jakości')

    # Wyświetlanie wyników
    plt.show()

    # Wczytanie obrazu (np. zdjęcie w skali szarości)
    nazwa_pliku = input("Podaj nazwę pliku obrazu (np. obraz.jpg): ")
    obraz = Image.open(nazwa_pliku).convert('L')  # Wczytywanie w skali szarości

    # Dodanie szumu "sól i pieprz"
    poziom_szumu = 0.05
    obraz_szum = dodaj_szum_sol_i_pieprz(obraz, poziom_szumu)

    # Wyświetlanie obrazu z szumem i jego histogramu
    pokaz_obraz_i_histogram(obraz, obraz_szum, "Obraz z szumem 'sól i pieprz'")

    # Zastosowanie filtrów
    maska_rozmiar = 3

    # Filtr uśredniający
    obraz_umediany = filtr_usredniajacy(obraz_szum, maska_rozmiar)
    pokaz_obraz_i_histogram(obraz_szum, obraz_umediany, "Filtr uśredniający")

    # Filtr medianowy
    obraz_medianowy = filtr_medianowy(obraz_szum, maska_rozmiar)
    pokaz_obraz_i_histogram(obraz_szum, obraz_medianowy, "Filtr medianowy")

    # Filtr minimum
    obraz_min = filtr_minimum(obraz_szum, maska_rozmiar)
    pokaz_obraz_i_histogram(obraz_szum, obraz_min, "Filtr minimum")

    # Filtr maksimum
    obraz_max = filtr_maksimum(obraz_szum, maska_rozmiar)
    pokaz_obraz_i_histogram(obraz_szum, obraz_max, "Filtr maksimum")

    ######
    # Wczytanie obrazu (np. zdjęcie w skali szarości)
    nazwa_pliku = input("Podaj nazwę pliku obrazu (np. obraz.jpg): ")
    obraz = Image.open(nazwa_pliku).convert('L')  # Wczytywanie w skali szarości

    # Rozmiar maski (możesz eksperymentować z różnymi wartościami)
    rozmiary_masek = [3, 5, 7]

    for rozmiar_maski in rozmiary_masek:
        # Zastosowanie filtra uśredniającego
        obraz_umediany = zastosuj_filtr_usredniajacy(obraz, rozmiar_maski)
        pokaz_obraz_i_histogram(obraz_umediany, f'Filtr uśredniający - {rozmiar_maski}x{rozmiar_maski}')

        # Zastosowanie filtra gaussowskiego
        obraz_gaussowski = zastosuj_filtr_gaussowski(obraz, rozmiar_maski)
        pokaz_obraz_i_histogram(obraz_gaussowski, f'Filtr gaussowski - {rozmiar_maski}x{rozmiar_maski}')

    ######
    # Wczytanie obrazu (np. zdjęcie w skali szarości)
    nazwa_pliku = input("Podaj nazwę pliku obrazu (np. obraz.jpg): ")
    obraz = Image.open(nazwa_pliku).convert('L')  # Wczytanie obrazu w odcieniach szarości

    # 1. Wykrywanie krawędzi - Sobel
    krawedzie_poziome, krawedzie_pionowe, krawedzie_ukosne = filtr_sobel(obraz)
    pokaz_obraz_i_histogram(krawedzie_poziome, 'Krawędzie poziome Sobel')
    pokaz_obraz_i_histogram(krawedzie_pionowe, 'Krawędzie pionowe Sobel')
    pokaz_obraz_i_histogram(krawedzie_ukosne, 'Krawędzie ukośne Sobel')

    # 2. Wyostrzanie obrazu za pomocą Laplasjanu
    wyostrzony_obraz = filtr_laplasjan(obraz)
    pokaz_obraz_i_histogram(wyostrzony_obraz, 'Wyostrzony obraz - Laplasjan')

    # 3. Unsharp Masking
    unsharp_obraz = unsharp_masking(obraz, sigma=1.0, amount=1.5)
    pokaz_obraz_i_histogram(unsharp_obraz, 'Wyostrzony obraz - Unsharp Masking')

    # 4. High Boost Filtering
    high_boost_obraz = high_boost_filtering(obraz, sigma=1.0, gain=1.5)
    pokaz_obraz_i_histogram(high_boost_obraz, 'Wyostrzony obraz - High Boost')


if __name__ == "__main__":
    main()
