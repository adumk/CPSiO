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
    """Wyrównywanie histogramu obrazu w skali szarości."""
    # Konwersja do skali szarości
    obraz_gray = obraz.convert('L')
    obraz_array = np.array(obraz_gray)

    # Histogram
    hist, bins = np.histogram(obraz_array.flatten(), bins=256, range=(0, 256))

    # Skumulowana funkcja rozkładu (CDF)
    cdf = hist.cumsum()

    # Normalizacja CDF do zakresu [0, 255]
    cdf_m = np.ma.masked_equal(cdf, 0)  # Maskujemy zera
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # Normalizacja
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # Zastępujemy maskę zerami

    # Mapowanie oryginalnych pikseli przez wyrównany CDF
    obraz_eq = cdf[obraz_array]

    # Przekształcenie z powrotem do obrazu
    return Image.fromarray(obraz_eq)


def rysuj_histogram(obraz, tytul):
    """Rysowanie histogramu obrazu."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)

    plt.hist(obraz_array.flatten(), bins=256, range=(0, 256), color='black', alpha=0.7)
    plt.title(tytul)
    plt.xlabel('Poziom szarości')
    plt.ylabel('Liczba pikseli')
    plt.grid(True)


def lokalne_wyrownanie_histogramu(obraz, rozmiar_okna=3):
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


def lokalne_poprawa_jakosci(obraz, rozmiar_okna=10):
    """Poprawa jakości obrazu oparta na lokalnych statystykach."""
    obraz_gray = obraz.convert('L')  # Przekształcamy obraz do odcieni szarości
    obraz_array = np.array(obraz_gray)

    # Parametry progowe
    C = 22.8
    k0 = k2 = 0
    k1 = k3 = 0.1

    # Globalne statystyki
    mG = np.mean(obraz_array)
    sigmaG = np.std(obraz_array)

    k, l = rozmiar_okna, rozmiar_okna
    wyniki = np.zeros_like(obraz_array)

    # Przechodzimy po obrazie w oknach
    for i in range(rozmiar_okna // 2, obraz_array.shape[0] - rozmiar_okna // 2):
        for j in range(rozmiar_okna // 2, obraz_array.shape[1] - rozmiar_okna // 2):
            okno = obraz_array[i - k // 2:i + k // 2 + 1, j - l // 2:j + l // 2 + 1]

            # Obliczanie lokalnych statystyk: średnia i odchylenie standardowe
            mSxy = np.mean(okno)
            sigmaSxy = np.std(okno)

            fxy = obraz_array[i, j]

            # Zastosowanie funkcji progowej
            if (k0 * mG <= mSxy <= k1 * mG) and (k2 * sigmaG <= sigmaSxy <= k3 * sigmaG):
                wartosc = C * fxy
            else:
                wartosc = fxy

            # Zapisujemy przeskalowaną wartość z ograniczeniem do zakresu [0, 255]
            wyniki[i, j] = np.clip(wartosc, 0, 255)

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


def filtr_sobel(obraz):
    """Wykrywanie krawędzi z użyciem filtra Sobela."""
    # Konwersja do odcieni szarości
    obraz_array = np.array(obraz.convert('L'))

    # Maski Sobela
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Konwolucja z maskami Sobela
    sobel_x = ndi.convolve(obraz_array.astype(float), Gx)
    sobel_y = ndi.convolve(obraz_array.astype(float), Gy)

    # Magnituda gradientu
    sobel_mag = np.hypot(sobel_x, sobel_y)
    sobel_x = np.hypot(sobel_x, sobel_x)
    sobel_y = np.hypot(sobel_y, sobel_y)
    sobel_mag = np.clip(sobel_mag / sobel_mag.max() * 255, 0, 255).astype(np.uint8)

    # Skalowanie wyników X i Y dla wizualizacji (z zachowaniem kontrastu)
    sobel_x_norm = np.clip((sobel_x - sobel_x.min()) / (sobel_x.max() - sobel_x.min()) * 255, 0, 255).astype(np.uint8)
    sobel_y_norm = np.clip((sobel_y - sobel_y.min()) / (sobel_y.max() - sobel_y.min()) * 255, 0, 255).astype(np.uint8)

    # Tworzenie obrazów wynikowych
    obraz_sobel_x = Image.fromarray(sobel_x_norm)
    obraz_sobel_y = Image.fromarray(sobel_y_norm)
    obraz_sobel_mag = Image.fromarray(sobel_mag)

    return obraz_sobel_x, obraz_sobel_y, obraz_sobel_mag

def filtr_laplasjan(obraz):
    """Zastosowanie filtra Laplasjanu do wyostrzania szczegółów."""
    obraz_array = np.array(obraz.convert('L'))
    laplasjan_maska = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Maska Laplasjanu

    # Konwolucja obrazu z maską Laplasjanu
    laplasjan_obraz = ndi.convolve(obraz_array, laplasjan_maska)

    # Wyostrzanie obrazu: oryginalny obraz + wyostrzona wersja (Laplasjan)
    wyostrzony_obraz = obraz_array + laplasjan_obraz

    # Konwersja: przycięcie wartości i zmiana typu danych
    wyostrzony_obraz = np.clip(wyostrzony_obraz, 0, 255).astype(np.uint8)

    # Konwersja z tablicy NumPy do obrazu PIL
    return Image.fromarray(wyostrzony_obraz), laplasjan_obraz


def unsharp_masking(obraz, sigma=5.0, amount=1):
    """Zastosowanie unsharp masking do wyostrzania obrazu."""
    obraz_array = np.array(obraz.convert('L'))

    # Rozmywanie obrazu (sztuczne rozmycie)
    rozmyty_obraz = ndi.gaussian_filter(obraz_array, sigma)

    rozmyty_obraz = np.array(rozmyty_obraz, dtype=np.float32) / 255.0
    obraz_array = np.array(obraz_array, dtype=np.float32) / 255.0

    # Subtrakcja rozmytego obrazu od oryginalnego (wyostrzanie)
    sharp_image = obraz_array + amount * (obraz_array - rozmyty_obraz)
    sharp_image = sharp_image * 255.0

    # Konwersja z tablicy NumPy do obrazu PIL
    return Image.fromarray(sharp_image)


def high_boost_filtering(obraz, sigma=5.0, gain=4.5):
    """Zastosowanie high boost filtering do wyostrzania obrazu."""
    obraz_array = np.array(obraz.convert('L'))

    # Rozmywanie obrazu (sztuczne rozmycie)
    rozmyty_obraz = ndi.gaussian_filter(obraz_array, sigma)

    rozmyty_obraz = np.array(rozmyty_obraz, dtype=np.float32) / 255.0
    obraz_array = np.array(obraz_array, dtype=np.float32) / 255.0

    # High boost filtering (zwiększanie szczegółów)
    high_boost_image = obraz_array + gain * (obraz_array - rozmyty_obraz)
    high_boost_image = high_boost_image * 255.0

    # Konwersja z tablicy NumPy do obrazu PIL
    return Image.fromarray(high_boost_image)


def cwiczenie5():
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

def cwiczenie6():
    # 1. Wczytywanie obrazu
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

def cwiczenie7():

    nazwy = [ 'chest-xray.tif', 'pollen-dark.tif', 'pollen-ligt.tif','pollen-lowcontrast.tif', 'pout.tif', 'spectrum.tif']


    for nazwa in nazwy:
        # Wczytywanie obrazów (np. obrazy zbyt ciemne i zbyt jasne)
        #nazwa_pliku = input("Podaj nazwę pliku obrazu (ciemny lub jasny): ")
        nazwa_pliku = nazwa
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

def cwiczenie8():
    # Wczytywanie obrazu (np. obrazy zbyt ciemne i zbyt jasne)
    #nazwa_pliku = input("Podaj nazwę pliku obrazu (ciemny lub jasny): ")
    nazwa_pliku = "hidden-symbols.tif"
    obraz = wczytaj_obraz(nazwa_pliku)

    rozmiar_okna_wyrownania = input("Podaj rozmiar okna wyrownania: ")
    rozmiar_okna_poprawy = input("Podaj rozmiar okna poprawy: ")

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
    obraz_wyrownany = lokalne_wyrownanie_histogramu(obraz, int(rozmiar_okna_wyrownania))

    # Wyświetlanie obrazu po lokalnym wyrównaniu histogramu
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(obraz_wyrownany, cmap='gray')
    plt.title(f'Obraz po lokalnym wyrównaniu histogramu z maską rozmiaru {rozmiar_okna_wyrownania}')
    plt.axis('off')

    # Rysowanie histogramu po wyrównaniu
    plt.subplot(1, 2, 2)
    rysuj_histogram(obraz_wyrownany, 'Histogram po wyrównaniu')

    # Wykonanie poprawy jakości na podstawie lokalnych statystyk
    obraz = wczytaj_obraz(nazwa_pliku)
    obraz_poprawiony = lokalne_poprawa_jakosci(obraz, int(rozmiar_okna_poprawy))

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

def cwiczenie9(nazwa_pliku):
    # Wczytanie obrazu (np. zdjęcie w skali szarości)
    #nazwa_pliku = input("Podaj nazwę pliku obrazu (np. obraz.jpg): ")
    obraz = Image.open(nazwa_pliku).convert('L')  # Wczytywanie w skali szarości

    # Dodanie szumu "sól i pieprz"
    poziom_szumu = 0.05
    obraz_szum = dodaj_szum_sol_i_pieprz(obraz, poziom_szumu)

    # Wyświetlanie obrazu z szumem i jego histogramu
    pokaz_obraz_i_histogram(obraz_szum, "Obraz z szumem 'sól i pieprz'")

    # Zastosowanie filtrów
    rozmiary = [3,5,7]


    for maska_rozmiar in rozmiary:
        # Filtr uśredniający
        obraz_umediany = filtr_usredniajacy(obraz_szum, maska_rozmiar)
        pokaz_obraz_i_histogram( obraz_umediany, f"Liniowy filtr uśredniający maska - {maska_rozmiar}")

        # Filtr medianowy
        obraz_medianowy = filtr_medianowy(obraz_szum, maska_rozmiar)
        pokaz_obraz_i_histogram( obraz_medianowy, f"Filtr medianowy maska - {maska_rozmiar}")

        # Filtr minimum
        obraz_min = filtr_minimum(obraz_szum, maska_rozmiar)
        pokaz_obraz_i_histogram(obraz_min, f"Filtr minimum maska - {maska_rozmiar}")

        # Filtr maksimum
        obraz_max = filtr_maksimum(obraz_szum, maska_rozmiar)
        pokaz_obraz_i_histogram( obraz_max, f"Filtr maksimum maska - {maska_rozmiar}")

def cwiczenie10(name):
    ######
    # Wczytanie obrazu (np. zdjęcie w skali szarości)
    #nazwa_pliku = input("Podaj nazwę pliku obrazu (np. obraz.jpg): ")
    nazwa_pliku = name
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

def cwiczenie11():
    cwiczenie11_1()
    #cwiczenie11_2()

def cwiczenie11_1():
    ######
    # Wczytanie obrazu (np. zdjęcie w skali szarości)
    obraz = Image.open('circuitmask.tif') # Wczytanie obrazu w odcieniach szarości

    # 1. Wykrywanie krawędzi - Sobel
    krawedzie_poziome, krawedzie_pionowe, krawedzie_ukosne = filtr_sobel(obraz)
    pokaz_obraz_i_histogram(krawedzie_poziome, 'Krawędzie poziome Sobel')
    pokaz_obraz_i_histogram(krawedzie_pionowe, 'Krawędzie pionowe Sobel')
    pokaz_obraz_i_histogram(krawedzie_ukosne, 'Krawędzie ukośne Sobel')

    obraz = Image.open('testpat1.png')  # Wczytanie obrazu w odcieniach szarości

    # 1. Wykrywanie krawędzi - Sobel
    krawedzie_poziome, krawedzie_pionowe, krawedzie_ukosne = filtr_sobel(obraz)
    pokaz_obraz_i_histogram(krawedzie_poziome, 'Krawędzie poziome Sobel')
    pokaz_obraz_i_histogram(krawedzie_pionowe, 'Krawędzie pionowe Sobel')
    pokaz_obraz_i_histogram(krawedzie_ukosne, 'Krawędzie ukośne Sobel')

    obraz = Image.open('blurry-moon.tif')  # Wczytanie obrazu w odcieniach szarości

    # 2. Wyostrzanie obrazu za pomocą Laplasjanu
    wyostrzony_obraz, elo = filtr_laplasjan(obraz)
    pokaz_obraz_i_histogram(wyostrzony_obraz, 'Wyostrzony obraz - Laplasjan')


def cwiczenie11_2():
    obraz = Image.open('text-dipxe-blurred.tif')  # Wczytanie obrazu w odcieniach szarości

    # 3. Unsharp Masking
    unsharp_obraz = unsharp_masking(obraz, sigma=5.0, amount=1)
    pokaz_obraz_i_histogram(unsharp_obraz, 'Wyostrzony obraz - Unsharp Masking')

    obraz = Image.open('text-dipxe-blurred.tif')  # Wczytanie obrazu w odcieniach szarości

    # 4. High Boost Filtering
    high_boost_obraz = high_boost_filtering(obraz, sigma=5.0, gain=4.5)
    pokaz_obraz_i_histogram(high_boost_obraz, 'Wyostrzony obraz - High Boost')


def cwiczenie12():
    #nazwa_pliku = input("Podaj nazwę pliku obrazu (np. obraz.jpg): ")
    obraz = Image.open("bonescan.tif")
    pokaz_obraz_i_histogram(obraz, 'Początkowy obraz')


    obraz, obraz_laplasjan = filtr_laplasjan(obraz)
    pokaz_obraz_i_histogram(obraz, 'Obraz + Laplasjan 3x3')

    poz,pio,uks = filtr_sobel(obraz)

    obraz = np.array(obraz.convert('L')) + np.array(uks.convert('L'))
    obraz = Image.fromarray(obraz)

    pokaz_obraz_i_histogram(obraz, 'Obraz po wykorzystaniu gradientu Sobela')

    obraz = filtr_usredniajacy(obraz, 5)
    pokaz_obraz_i_histogram(obraz, 'Obraz po filtracji uśredniającej z maską 5x5')



    obraz = np.array(obraz.convert('L'), dtype=np.float32) / 255.0
    obraz_laplasjan = np.array(obraz_laplasjan, dtype=np.float32) / 255.0
    obraz = obraz * obraz_laplasjan
    obraz = obraz * 255.0
    obraz = Image.fromarray(obraz)
    pokaz_obraz_i_histogram(obraz, 'Iloczyn obrazu i Laplasjanu')

    obraz = np.array(obraz.convert('L'), dtype=np.float32) / 255.0
    obraz_wstepny = np.array(Image.open("bonescan.tif").convert('L'), dtype=np.float32) / 255.0
    obraz = obraz + obraz_wstepny
    obraz = obraz * 255.0
    obraz = Image.fromarray(obraz)
    pokaz_obraz_i_histogram(obraz, 'Suma obraz wstepnego i aktualnego')

    obraz = przeksztalcenie_gamma(obraz, 1, 0.5)
    pokaz_obraz_i_histogram(obraz, 'Transformacja potegowa c=1 y=0.5')


def main():
    #cwiczenie5()
    #cwiczenie6()
    #cwiczenie7()
    #while True:
    #    cwiczenie8()
    # nazwy = ['cboard_pepper_only.tif', 'cboard_salt_only.tif', 'cboard_salt_pepper.tif']
    # for name in nazwy:
    #     cwiczenie9(name)

    # nazwy = ['characters_test_pattern.tif', 'zoneplate.tif']
    # for name in nazwy:
    #     cwiczenie10(name)
    #cwiczenie10()
    #cwiczenie11()
    cwiczenie12()

if __name__ == "__main__":
    main()
