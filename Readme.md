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

### Konteks Bisnis
Dalam industri perhotelan, ulasan pelanggan memainkan peran kritis dalam membentuk citra dan reputasi bisnis. Namun, volume besar ulasan online membuat analisis manual menjadi tidak efisien dan tidak praktis.

### Problem Statements
Berdasarkan latar belakang dan tantangan bisnis di atas, berikut adalah masalah utama yang akan diselesaikan:

- Bagaimana hotel dapat secara efisien dan akurat memahami sentimen pelanggan dari ribuan ulasan online?
- Bagaimana mengubah data ulasan mentah menjadi wawasan strategis yang dapat meningkatkan kualitas layanan?

### Goals
- Mengembangkan sistem otomatis untuk menganalisis sentimen ulasan hotel guna:
  - Mengidentifikasi area perbaikan layanan secara cepat dan presisi
  - Membantu manajemen hotel membuat keputusan berbasis data
  - Meningkatkan kepuasan pelanggan melalui umpan balik tepat waktu

- Merancang model machine learning yang dapat:

  - Mengklasifikasikan sentimen dengan akurasi tinggi
  - Memberikan insight yang dapat ditindaklanjuti oleh tim manajemen

### Manfaat Bisnis

1. Analisis Cepat Umpan Balik Pelanggan
- Memproses ribuan ulasan dalam waktu singkat
- Mendeteksi tren sentimen secara real-time

2. Peningkatan Kualitas Layanan
- Identifikasi cepat aspek-aspek yang perlu diperbaiki
- Fokus pada area yang secara konsisten menerima umpan balik negatif

3. Strategi Pemasaran Berbasis Data
- Memahami persepsi pelanggan tentang layanan hotel
- Merancang kampanye pemasaran yang lebih tepat sasaran

4. Keunggulan Kompetitif
- Respon cepat terhadap kebutuhan pelanggan
- Membangun reputasi sebagai hotel yang peduli dan responsif

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

### Informasi Dataset
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

### Analisis Kuantitatif Dataset
1. Jumlah Data
- Total Baris: 20491 Baris
- Total Kolom: 2 (Review dan Rating)

2. Statistik Deskriptif Kolom
- Kolom Review
  - Tipe Data: String
  - Deskripsi: Kalimat/sentimen/pendapat/kritik yang ditulis oleh individu

- Kolom Rating
  - Tipe Data: Integer
  - Rentang Nilai: 1-5
  - Deskripsi: Perasaan yang dirasakan oleh individu yang digambarkan dengan skala angka

3. Kondisi Data
- Pemeriksaan Kualitas Data
  - Missing Values: 0
  - Duplikat: 0
  - Distribusi Rating

| Rating | Jumlah Review|
|--------| -------------|
| 1      | 9054         |
| 2      | 6039         |
| 3      | 2184         |
| 4      | 1793         |
| 5      | 1421         |

  - Tampilan Dataset

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

### Teknik Penanganan Data

1. Pembersihan Data

Metode pembersihan teks yang dilakukan:
- Menghapus tag HTML
- Menghapus URL
- Menghapus emoji
- Menghapus angka
- Menghapus tanda baca
- Mengonversi teks ke huruf kecil
- Menghapus stopwords

2. Transformasi Data

Konversi rating menjadi label sentimen:

- Positif (1): Rating 3, 4, dan 5
- Negatif (0): Rating 1 dan 2

3. Pembagian Dataset

Teknik pembagian data menggunakan stratified sampling:

- Training Set: 70% dari total data
- Validation Set: 15% dari total data
- Test Set: 15% dari total data

Tujuan Pembagian:

- Memastikan distribusi label sentimen seimbang
- Mencegah bias dalam proses pelatihan model
- Menyediakan data independen untuk evaluasi

4. Tokenisasi

Proses mengubah teks menjadi format yang dapat diproses model:

- Menggunakan BERT Tokenizer
- Maksimum panjang token: 128
- Padding dan truncation dilakukan untuk konsistensi

5. Pertimbangan Teknis

Metode pembersihan dirancang untuk:

- Mengurangi noise dalam data
- Meningkatkan kualitas input model
- Memfokuskan analisis pada konten utama review

6. Membaca dataset yang sudah bersih ke DataFrame pandas

Pada bagian ini akan ditampilkan dataset yang sudah bersih, lalu tampilannya akan seperti tabel 3.

Tabel 3. Tampilan dataset yang sudah bersih bentuk _DataFrame pandas_.  

|   | Review                                               | Rating   |sentiment_encoded|  sentiment_label | 
| - | ---------------------------------------------------- | ---------|-----------------|------------------|
| 0 | nice hotel expensive parking got good deal sta...    | 4        |        1        |Positive          |
| 1 | ok nothing special charge diamond member hilto...    | 2        |        0        |Negative          |
| 2 | nice rooms experience hotel monaco seattle goo...    | 3        |        1        |Positive          |
| 3 | unique great stay wonderful time hotel monaco ...    | 5        |        1        |Positive          |
| 4 | great stay great stay went seahawk game awesom...    | 5        |        1        |Positive          |

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

Dapat dilihat dari **Confusion Matrix** bahwa model menunjukkan performa yang baik dengan nilai True Positive mencapai 4028 dan True Negative mencapai 592. Jumlah False Positive dan False Negative relatif rendah, masing-masing 179 dan 119.

### Classification Report
*** 
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

Dari **Model Accuracy Plot**, kita dapat melihat bahwa model mendapatkan performa yang baik dalam train dengan akurasi 97% dan validasi 94%. Performa baik ini juga dibuktikan dengan evaluasi model menggunakan data test yang disiapkan. Dari data test dapat dibuat **Classification Report** yang ada di tabel 4. Dari tabel 4 kita dapat melihat skor dari model terhadap 3 metrik yang dihasilkan oleh classification report. Dilihat dari hasil, dapat disimpulkan bahwa model yang dibuat Good Fit.

Referensi:  
  [1] [Alam, M. H., Ryu, W.-J., Lee, S., 2016. Joint multi-grain topic sentiment: modeling semantic aspects for online reviews. Information Sciences 339, 206–223.](https://www.sciencedirect.com/science/article/abs/pii/S0020025516000153)