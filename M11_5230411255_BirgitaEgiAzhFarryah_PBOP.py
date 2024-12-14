import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

#Preprocessing
def preprocess_data(nama_file):
    try:
        # Memuat dataset
        data = pd.read_excel(nama_file)

        # Mengonversi semua nilai non-numerik menjadi NaN
        data = data.apply(pd.to_numeric, errors='coerce')

        # Memeriksa nilai yang hilang dan mengisinya
        if data.isnull().sum().sum() > 0:
            print("Nilai hilang atau non-numerik terdeteksi. Mengisi dengan nilai rata-rata.")
            data.fillna(data.mean(), inplace=True)

        # Memisahkan fitur dan target
        fitur = data.iloc[:, :-1]
        target = data.iloc[:, -1] 

        # Menangani nilai hilang atau non-numerik di kolom target
        if target.isnull().sum() > 0:
            print("Nilai hilang di kolom target terdeteksi. Mengisi dengan nilai default (0).")
            target.fillna(0, inplace=True)

        # Memastikan kolom target adalah integer untuk klasifikasi
        target = target.astype(int)

        if fitur.empty or target.empty:
            raise ValueError("Dataset tidak memiliki data numerik yang cukup untuk fitur atau target.")

        return fitur, target

    except FileNotFoundError:
        raise FileNotFoundError("File tidak ditemukan. Pastikan nama file dan path sudah benar.")
    except Exception as e:
        raise Exception(f"Terjadi kesalahan saat preprocessing: {e}")

#Naive Bayes
def analisis_naive_bayes(fitur, target):
    try:
        X_latih, X_uji, y_latih, y_uji = train_test_split(fitur, target, test_size=0.2, random_state=42)

        model = GaussianNB()

        model.fit(X_latih, y_latih)

        prediksi = model.predict(X_uji)

        akurasi = accuracy_score(y_uji, prediksi)
        laporan = classification_report(y_uji, prediksi)

        return akurasi, laporan
    except Exception as e:
        raise Exception(f"Terjadi kesalahan saat analisis Naive Bayes: {e}")

# Fungsi Analisis Neural Network
def analisis_neural_network(fitur, target):
    try:

        X_latih, X_uji, y_latih, y_uji = train_test_split(fitur, target, test_size=0.2, random_state=42)


        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

    
        model.fit(X_latih, y_latih)

    
        prediksi = model.predict(X_uji)

        akurasi = accuracy_score(y_uji, prediksi)
        laporan = classification_report(y_uji, prediksi)

        return akurasi, laporan
    except Exception as e:
        raise Exception(f"Terjadi kesalahan saat analisis Neural Network: {e}")

if __name__ == "__main__":
    while True:
        print("\n=== MENU ANALISIS DATA ===")
        print("1. Preprocessing Data")
        print("2. Analisis Naive Bayes")
        print("3. Analisis Neural Network")
        print("4. Keluar")
        
        pilihan = input("Pilih menu: ")
        
        if pilihan == '1':
            nama_file = input("Masukkan nama file dataset : ")
            try:
                fitur, target = preprocess_data(nama_file)
                print("Preprocessing selesai! Data siap digunakan.")
            except Exception as e:
                print(f"Error: {e}")

        elif pilihan == '2':
            try:
                if 'fitur' not in locals() or 'target' not in locals():
                    print("Silakan lakukan preprocessing terlebih dahulu!")
                    continue

                akurasi, laporan = analisis_naive_bayes(fitur, target)
                print("Analisis Naive Bayes selesai! Berikut hasilnya:")
                print(f"Akurasi: {akurasi * 100:.2f}%")
                print("Laporan Klasifikasi:")
                print(laporan)
            except Exception as e:
                print(f"Error: {e}")

        elif pilihan == '3':
            try:
                if 'fitur' not in locals() or 'target' not in locals():
                    print("Silakan lakukan preprocessing terlebih dahulu!")
                    continue

                akurasi, laporan = analisis_neural_network(fitur, target)
                print("Analisis Neural Network selesai! Berikut hasilnya:")
                print(f"Akurasi: {akurasi * 100:.2f}%")
                print("Laporan Klasifikasi:")
                print(laporan)
            except Exception as e:
                print(f"Error: {e}")

        elif pilihan == '4':
            print("Keluar dari program.")
            break

        else:
            print("Pilihan tidak valid. Silakan coba lagi.")
