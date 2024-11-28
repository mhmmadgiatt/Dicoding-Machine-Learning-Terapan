# Laporan Proyek Machine Learning - Muhammad Giat
***
## Domain Proyek
***
### Latar Belakang
Hotel memiliki peran penting dalam dunia perjalanan, dan dengan meningkatnya akses informasi, cara baru untuk memilih hotel terbaik pun muncul. Salah satu cara tersebut adalah dengan membaca ulasan (review) yang ditinggalkan oleh tamu hotel.

Ulasan-ulasan ini mengandung sentimen, opini, kritik, dan masukan yang sangat berharga bagi hotel sebagai indikator kepuasan pelanggan. Secara umum, sentimen ini dapat dikategorikan menjadi positif dan negatif, atau lebih spesifik lagi berdasarkan aspek tertentu dari pengalaman pelanggan seperti kebersihan, layanan, lokasi, dan kenyamanan.

Namun, menganalisis sentimen dari ribuan ulasan secara manual memakan waktu dan tenaga. Oleh karena itu, diperlukan sebuah algoritma yang dapat memprediksi sentimen dari ulasan tersebut secara otomatis. Dalam proyek ini, saya mengembangkan sebuah model Natural Language Processing (NLP) untuk memprediksi sentimen ulasan berdasarkan rating yang diberikan tamu hotel. Model ini akan dilatih dan diuji menggunakan dataset Trip Advisor Hotel Reviews yang berisi 20,491 ulasan [1].
## Business Understanding
***
### Problem Statements
Berdasarkan latar belakang di atas, berikut adalah masalah yang akan diselesaikan dalam proyek ini:
- Bagaimana melakukan pra-pemrosesan teks pada data ulasan hotel agar siap digunakan untuk melatih model?
- Bagaimana merancang arsitektur model deep learning untuk memprediksi sentimen berdasarkan rating ulasan?

### Goals
- Melakukan pra-pemrosesan data teks dengan baik agar siap digunakan untuk melatih model.
- Merancang dan mengimplementasikan model deep learning untuk memprediksi sentimen berdasarkan data ulasan hotel.

### Solution Statements
Solusi yang dapat dilakukan sebagai berikut:
1. Melakukan Pra-pemrosesan Data

- Menghapus karakter tidak penting seperti tag HTML, URL, emoji, angka, tanda baca, dan spasi berlebih.
- Normalisasi teks dengan mengonversinya ke huruf kecil.
- Menghapus stopwords menggunakan pustaka NLTK.
- Mengonversi rating menjadi label sentimen:
  - Positif: Rating 3, 4, dan 5.
  - Negatif: Rating 1 dan 2.

2. Membangun Model Deep Learning
- Memanfaatkan transformer-based model seperti BERT untuk representasi teks dan klasifikasi sentimen.
- Menggunakan tokenizer BERT untuk memproses teks dengan metode tokenisasi yang sesuai dengan arsitektur model.
- Menerapkan kelas Dataset untuk mempermudah pengelolaan data dalam bentuk tensor yang digunakan oleh PyTorch.
- Membagi dataset menjadi tiga bagian: pelatihan, validasi, dan pengujian, dengan proporsi yang sesuai.

3. Pelatihan Model
- Melatih model BERT menggunakan pustaka PyTorch dengan parameter yang disesuaikan, seperti learning rate dan jumlah epoch.
- Menggunakan algoritma optimisasi AdamW.

4. Evaluasi Model
- Mengukur performa model pada data validasi dan pengujian menggunakan metrik berikut:
- Akurasi: Persentase prediksi yang benar terhadap total data.
- Precision, Recall, dan F1-score: Metrik tambahan untuk menilai keseimbangan performa model pada masing-masing kelas sentimen.
- Menggunakan classification report untuk memberikan gambaran detail performa model pada kelas "Positif" dan "Negatif".

5. Pengujian Model
- Menggunakan data pengujian untuk mengevaluasi performa akhir model.
- Menghitung akurasi keseluruhan pada data pengujian untuk memberikan indikator performa model pada data baru.

## Data Understanding
***
Dataset yang digunakan dapat diakses menggunakan [Kaggle](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews?resource=download)  
Informasi dari dataset dapat dirangkum sebagai berikut:

Tabel 1. Rangkuman informasi Dataset    

| Jenis                  | Keterangan                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Sumber                 | [Kaggle Dataset: Trip Advisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews?resource=download)  |
| Lisensi                | [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)                                                   |
| Kategori               | Ekonomi dan Bisnis                                                                                                            |
| Jenis & Ukuran berkas  | CSV (14.97MB)                                                                                                      |  
***
Pada berkas yang diunduh berisi 1 berkas dengan jenis format _.csv_ yang bernama tripadvisor_hotel_reviews.csv.File csv ini memiliki 2 kolom yang bertipe data string dan integer. Untuk penjelasan mengenai variabel pada dataset sebagai berikut:  
- Review: kalimat/sentimen/pendapat/kritik yang ditulis oleh individu
- Rating: perasaan yang dirasakan oleh individu yang di gambarkan dengan skala angka 1-5.

Tabel 2. Tampilan dataset awal dalam bentuk _DataFrame pandas_.  

|   | Review                                               | Rating   |
| - | ---------------------------------------------------- | ---------|
| 0 | nice hotel expensive parking got good deal sta...    | 4        |  
| 1 | ok nothing special charge diamond member hilto...    | 2        |  
| 2 | nice rooms experience hotel monaco seattle goo...    | 3        |    
| 3 | unique great stay wonderful time hotel monaco ...    | 5        | 
| 4 | great stay great stay went seahawk game awesom...    | 5        |

***

## Data Preparation

### Langkah-langkah pra-pemrosesan data
1. Mendownload Dataset
- Dataset digunakan dalam proyek ini adalah tripadvisor_hotel_reviews.csv, yang berisi ulasan dan rating dari pengguna mengenai hotel di Tripadvisor. Dataset ini dibaca dari file CSV menggunakan Pandas.
2. Mengecek Data
- Setelah memuat data, langkah pertama adalah memeriksa informasi umum mengenai dataset untuk memastikan kualitas data.
3. Pembersihan Teks
Data teks dari ulasan perlu dibersihkan agar model dapat memprosesnya dengan lebih baik. Berikut adalah langkah-langkah pembersihan yang dilakukan:
- Menghapus tag HTML
- Menghapus URL
- Menghapus emoji
- Menghapus angka
- Menghapus tanda baca
- Mengubah teks menjadi huruf kecil
- Menghapus stopwords (kata umum yang tidak memberikan makna penting dalam analisis)
4. Membaca dataset yang sudah bersih ke DataFrame pandas
- Pada bagian ini akan ditampilkan dataset yang sudah bersih, lalu tampilannya akan seperti tabel 3.

Tabel 3. Tampilan dataset yang sudah bersih bentuk _DataFrame pandas_.  

|   | Review                                               | Rating   |sentiment_encoded|  sentiment_label | 
| - | ---------------------------------------------------- | ---------|-----------------|------------------|
| 0 | nice hotel expensive parking got good deal sta...    | 4        |        1        |Positive          |
| 1 | ok nothing special charge diamond member hilto...    | 2        |        0        |Negative          |
| 2 | nice rooms experience hotel monaco seattle goo...    | 3        |        1        |Positive          |
| 3 | unique great stay wonderful time hotel monaco ...    | 5        |        1        |Positive          |
| 4 | great stay great stay went seahawk game awesom...    | 5        |        1        |Positive          |

### Pemetaan Sentimen

Pada tahap ini, kita mengonversi rating menjadi label sentimen biner:
- Rating lebih dari atau sama dengan 3 dianggap sebagai ulasan positif (1).
- Rating kurang dari 3 dianggap sebagai ulasan negatif (0).

### Pembagian Data
Dataset dibagi menjadi tiga bagian utama: training set, validation set, dan test set. Pembagian dilakukan dengan teknik stratified sampling untuk memastikan distribusi label sentimen tetap seimbang di semua subset. Data training digunakan untuk melatih model, sementara data validasi digunakan untuk mengukur performa model selama proses pelatihan. Data test digunakan sebagai evaluasi akhir untuk menilai kemampuan model pada data yang benar-benar baru.
```bash
X = combined_df['cleaned_review'].values
y = combined_df['sentiment_encoded'].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
```

### Tokenisasi
Pada tahap ini, data teks ulasan yang telah disiapkan diubah menjadi format yang dapat diproses oleh model BERT. Proses ini terdiri dari beberapa tahapan penting, masing-masing dengan parameter yang disesuaikan untuk menghasilkan representasi data yang optimal.

1. Tokenisasi dengan Tokenizer Pra-Latihan BERT
Setiap ulasan teks diproses menggunakan tokenizer pra-latihan BERT (bert-base-uncased) dari pustaka transformers. Tokenizer ini memecah teks menjadi unit kecil (subwords) atau token, memastikan bahwa setiap kata direpresentasikan sesuai dengan vokabulari model BERT. Hasil tokenisasi mencakup:

- input_ids: Representasi indeks token berdasarkan vokabulari BERT.
- attention_mask: Masking biner (0 dan 1) yang menunjukkan token mana yang harus diperhatikan model (1) dan mana yang diabaikan (0, seperti padding).

Parameter Tokenizer:

- Maksimum Panjang Token: 128 token. Ini ditetapkan untuk menjaga efisiensi komputasi sambil tetap menangkap konteks ulasan. Ulasan yang lebih pendek dari panjang ini akan dipenuhi (padded), sementara ulasan yang lebih panjang akan dipotong (truncated).
- Do Lowercase: Karena menggunakan model uncased, semua huruf diubah menjadi huruf kecil untuk konsistensi dengan vokabulari BERT.

2. Konversi ke Format PyTorch Dataset
Setelah tokenisasi, data hasil tokenisasi diubah ke dalam format PyTorch Dataset, sehingga dapat dengan mudah dikelola selama proses pelatihan. Setiap sampel dalam dataset mencakup:

- input_ids: Indeks token.
- attention_mask: Masking biner.
- labels: Label sentimen ulasan (positif = 1, negatif = 0).

3. Pembagian Data dan Loader
Dataset dibagi menjadi tiga subset (training, validation, dan test) menggunakan metode stratifikasi untuk menjaga distribusi label yang seimbang. Setelah itu, masing-masing subset dimasukkan ke dalam DataLoader untuk digunakan selama pelatihan dan evaluasi.

Parameter DataLoader:

- Batch Size: 16. Batch kecil digunakan untuk meminimalkan kebutuhan memori GPU sambil memastikan stabilitas proses pelatihan.
- Shuffling: Diaktifkan untuk dataset training agar model tidak belajar pola tertentu berdasarkan urutan data.

Dengan mengikuti langkah-langkah ini, dataset yang telah diproses menjadi siap digunakan oleh model BERT untuk tugas klasifikasi sentimen, memastikan bahwa setiap teks ulasan diterjemahkan ke dalam format numerik yang sesuai dengan spesifikasi model.

## Modeling
***
### Pelatihan Model BERT
Proses pelatihan model dilakukan menggunakan model BERT for Sequence Classification, yang merupakan varian BERT yang dirancang untuk tugas klasifikasi teks. Model ini memanfaatkan representasi berbasis transformer yang mendalam untuk memahami konteks dalam teks, menjadikannya sangat efektif untuk tugas analisis sentimen.

### Tahapan Pemodelan
1. Inisialisasi Model dan Tokenizer

Model dasar yang digunakan adalah pre-trained BERT dari pustaka Hugging Face Transformers, dengan arsitektur khusus untuk klasifikasi biner. Tokenizer BERT pra-latihan digunakan untuk memproses data teks menjadi format numerik berupa input_ids dan attention_mask. Input ini memastikan bahwa model dapat memahami struktur dan hubungan antar kata dalam teks.

2. Pembekuan Parameter Pra-latihan (Fine-tuning)

Parameter pre-trained BERT di-fine-tune untuk tugas analisis sentimen. Ini dilakukan dengan membiarkan lapisan-lapisan awal BERT tetap utuh, sementara lapisan klasifikasi di bagian akhir dilatih untuk memahami pola khusus dari data ulasan.

3. Pengaturan Hyperparameter

Hyperparameter yang digunakan selama proses pelatihan adalah:
- Learning Rate: 5e-5
Nilai ini dipilih berdasarkan rekomendasi umum untuk fine-tuning model berbasis BERT, memastikan pelatihan berlangsung stabil tanpa kehilangan informasi dari model pra-latihan.
- Batch Size: 16
Ukuran batch dipilih untuk mengakomodasi keterbatasan memori GPU, sambil tetap memberikan pembaruan parameter yang efektif.
- Epochs: 3
Pelatihan dilakukan selama tiga iterasi penuh pada data training, sesuai dengan praktik terbaik dalam pelatihan model BERT. Ini dirancang untuk menghindari overfitting pada dataset kecil.
- Optimizer: AdamW
Digunakan karena kemampuan penanganan gradien yang baik pada model berbasis transformer. Weight decay diterapkan untuk mengontrol overfitting.
- Scheduler: Linear Learning Rate Scheduler
Digunakan untuk menurunkan learning rate secara bertahap selama pelatihan, menghindari perubahan drastis pada model saat mendekati konvergensi.

4. Pelatihan Model

Proses pelatihan dilakukan dengan pendekatan mini-batch gradient descent. Pada setiap iterasi:

- Loss function yang digunakan adalah Binary Cross-Entropy Loss, sesuai dengan tugas klasifikasi biner.
- Model menerima tokenized input (termasuk attention_mask) dan label untuk menghitung prediksi.
- Gradien dihitung berdasarkan loss, kemudian dilakukan pembaruan parameter menggunakan optimizer AdamW.

5. Validasi pada Setiap Epoch

Setelah setiap epoch, model dievaluasi menggunakan data validasi. Metode ini memastikan performa model dapat dipantau, sehingga jika terjadi overfitting, pelatihan dapat dihentikan atau dioptimalkan lebih lanjut.

6. Checkpoint dan Early Stopping

Untuk mencegah overfitting, sistem menyimpan model dengan performa validasi terbaik sebagai checkpoint. Selain itu, early stopping dapat diterapkan jika tidak ada peningkatan performa signifikan setelah sejumlah epoch tertentu.

#### Parameter yang Digunakan
- Pretrained Model: BERT (base, uncased)
- Optimizer: AdamW
- Batch Size: 16
- Learning Rate: 5e-5
- Epochs: 3
- Loss Function: Binary Cross-Entropy Loss
- Scheduler: Linear Learning Rate Decay

#### Output Akhir
Hasil dari proses pelatihan adalah model BERT yang terlatih untuk tugas klasifikasi sentimen. Model ini mampu memberikan prediksi dengan akurasi tinggi pada data validasi, menjadikannya siap untuk diuji lebih lanjut pada data test.

## Evaluasi
***
Pada tahap evaluasi, performa model diukur menggunakan beberapa metrik evaluasi yang relevan dengan tugas klasifikasi biner. Pemilihan metrik ini disesuaikan dengan tujuan analisis sentimen, yaitu membedakan sentimen positif dan negatif secara akurat.

### Metrik Evaluasi yang Digunakan
1. Akurasi (Accuracy)

Akurasi digunakan untuk mengukur persentase prediksi model yang benar dibandingkan dengan seluruh prediksi.
- Formula:

$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$

Akurasi memberikan gambaran umum performa model, tetapi kurang efektif jika terdapat ketidakseimbangan kelas dalam data.

2. Precision

Precision mengukur proporsi prediksi positif yang benar. Metrik ini penting jika kesalahan pada prediksi positif lebih berdampak signifikan.
- Formula:

$\text{Precision} = \frac{TP}{TP+FP}$

3. Recall (Sensitivity)

Recall mengukur seberapa baik model dapat mendeteksi sampel positif yang sebenarnya.
- Formula:

$\text{Recall} = \frac{TP}{TP+FN}$

4. F1-Score

F1-Score adalah rata-rata harmonis antara precision dan recall, memberikan keseimbangan antara kedua metrik.
- Formula:

$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ 

Dengan:
- TP: True Positive
- TN: True Negative
- FP: False Positive
- FN: False Negative

### Model Accuracy Plot
*** 

![model_accuracy](https://raw.githubusercontent.com/mhmmadgiatt/Dicoding-Machine-Learning-Terapan/main/img/plot_performances.png)
 
Dapat dilihat dari **Model Accuracy Plot** bahwa setelah epochs ke-2, model menunjukkan performa yang baik dengan nilai akurasi train mencapai sekitar 97% dan akurasi validasi sekitar 94%.

### Confusion Matrix
*** 

![Confusion Matrix](https://raw.githubusercontent.com/mhmmadgiatt/Dicoding-Machine-Learning-Terapan/main/img/confusion_matrix.png) 

Penulis juga menguji model dengan data _test_ yang sebelumnya sudah dipisahkan dengan hasil seperti berikut tabel 4.  

Tabel 4. _Classification Report_

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| Negative     | 0.83      | 0.77   | 0.80     | 771     |
| Positive     | 0.96      | 0.97   | 0.96     | 4747    |


|               |      |      |      | support |
| ------------- | ---- | ---- | ---- | ------- |
| accuracy      |      |      | 0.94 | 4918    |
| macro avg     | 0.90 | 0.87 | 0.88 | 4918    |
| weighted avg  | 0.94 | 0.94 | 0.94 | 4918    |

Terlihat bahwa model bekerja dengan baik, dari dua label yang diprediksi didapat seluruh nilai di atas 70%, dengan akurasi test sebesar 94%.

- Precision
  - Negative dari 771 data yang model prediksi, 83% diprediksi memiliki sentimen negatif.

  - Positive dari 4747 data yang model prediksi, 96% diprediksi memiliki sentimen positif.
  
- Recall
  - Negative dari 83% yang diprediksi memiliki sentimen negatif, hanya 77% menghasilkan benar.

  - Positive dari 96% yang diprediksi memiliki sentimen positif, hanya 97% menghasilkan benar.  
  
- F1-Score
F1-Score Dari kedua label, dapat dilihat bahwa model menghasilkan performa yang baik, terutama pada label Positive karena hampir mendekati 100%, untuk label Negative mendapatkan nilai 80% yang masih cukup baik.

- Kesimpulan
Kesimpulan Dari **Model Accuracy Plot** kita dapat melihat bahwa model mendapatkan performa yang baik dalam train dengan akurasi 99% dan validasi 90%, performa baik ini juga dibuktikan dengan evaluasi model menggunakan data test yang disiapkan. Dari data test dapat dibuat classification report yang ada di tabel 4. Dari tabel 4 kita dapat melihat skor dari model terhadap 3 metrik yang di generate oleh classification report, dilihat dari hasil dapat disimpulkan bahwa model yang dibuat Good Fit.

Referensi:  
  [1] [Alam, M. H., Ryu, W.-J., Lee, S., 2016. Joint multi-grain topic sentiment: modeling semantic aspects for online reviews. Information Sciences 339, 206–223.](https://www.sciencedirect.com/science/article/abs/pii/S0020025516000153)