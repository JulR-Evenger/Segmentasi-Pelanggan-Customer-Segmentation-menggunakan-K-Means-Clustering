# Segmentasi Pelanggan (Customer Segmentation) menggunakan K-Means Clustering

![Customer Segmentation Banner](https://raw.githubusercontent.com/user-attachments/assets/b343818e-01fd-4b08-be64-67252033c415)

## Ringkasan Proyek (TL;DR)
Proyek ini bertujuan untuk melakukan segmentasi pelanggan sebuah pusat perbelanjaan berdasarkan pola pendapatan dan pengeluaran mereka. Dengan menggunakan algoritma **K-Means Clustering**, pelanggan dikelompokkan ke dalam **5 segmen** yang berbeda dan dapat ditindaklanjuti. Hasil segmentasi ini memberikan wawasan berharga bagi tim pemasaran untuk merancang strategi yang lebih personal dan efektif.

**Tags:** `Python`, `Scikit-Learn`, `Pandas`, `Matplotlib`, `Seaborn`, `Machine Learning`, `Clustering`, `K-Means`, `Analisis Data`

---

## Daftar Isi
1. [Latar Belakang Bisnis](#latar-belakang-bisnis)
2. [Tujuan Proyek](#tujuan-proyek)
3. [Dataset](#dataset)
4. [Alur Kerja Proyek (Workflow)](#alur-kerja-proyek)
5. [Hasil & Interpretasi Cluster](#hasil--interpretasi-cluster)
6. [Kesimpulan & Rekomendasi Bisnis](#kesimpulan--rekomendasi-bisnis)

---

### Latar Belakang Bisnis
Dalam dunia ritel yang kompetitif, memahami perilaku pelanggan adalah kunci untuk bertahan dan bertumbuh. Pendekatan pemasaran "satu untuk semua" seringkali tidak efektif. Dengan melakukan segmentasi, perusahaan dapat mengidentifikasi kelompok-kelompok pelanggan yang berbeda dan menyesuaikan produk, layanan, serta kampanye pemasaran agar sesuai dengan kebutuhan dan keinginan masing-masing segmen.

### Tujuan Proyek
* Menganalisis dan memvisualisasikan data pelanggan untuk menemukan pola tersembunyi.
* Menentukan jumlah cluster (segmen) yang optimal menggunakan **Metode Elbow**.
* Mengelompokkan pelanggan ke dalam segmen-segmen yang bermakna menggunakan K-Means.
* Memberikan interpretasi dan profil untuk setiap segmen pelanggan yang terbentuk.

### Dataset
Dataset yang digunakan (`customer.csv`) berisi informasi dasar pelanggan, dengan fokus utama pada dua fitur untuk clustering:
* **Annual Income (k$)**: Pendapatan tahunan pelanggan dalam ribuan dolar.
* **Spending Score (1-100)**: Skor yang diberikan oleh mal berdasarkan perilaku belanja pelanggan (nilai lebih tinggi berarti lebih sering berbelanja).

### Alur Kerja Proyek (Workflow)
1.  **Eksplorasi Data (EDA):** Melakukan analisis deskriptif dan visualisasi untuk memahami distribusi data, seperti distribusi usia, pendapatan, dan skor pengeluaran.
2.  **Penentuan Jumlah Cluster Optimal (K):**
    * Menggunakan **Metode Elbow** untuk menemukan jumlah cluster yang paling optimal.
    * Dari plot, terlihat bahwa "siku" terbentuk pada **K=5**, yang menunjukkan bahwa 5 adalah jumlah segmen yang paling sesuai untuk data ini.
3.  **Pemodelan K-Means:**
    * Algoritma K-Means dilatih pada data dengan menggunakan 5 cluster.
    * Fitur yang digunakan adalah 'Annual Income' dan 'Spending Score'.
4.  **Visualisasi Cluster:** Hasil clustering divisualisasikan dalam bentuk *scatter plot* untuk melihat pemisahan antar segmen secara jelas.

### Hasil & Interpretasi Cluster
Model berhasil mengidentifikasi 5 segmen pelanggan yang berbeda, masing-masing dengan karakteristik unik:

![Visualisasi Cluster Pelanggan](https://raw.githubusercontent.com/user-attachments/assets/b624773c-a99f-4315-99d7-83d8009d18e4)

* **Cluster 0 (Hijau) - Target Utama:** Pelanggan dengan pendapatan tinggi dan skor belanja tinggi. Mereka adalah target paling berharga.
* **Cluster 1 (Biru) - Hati-hati & Kaya:** Pelanggan dengan pendapatan tinggi namun skor belanja rendah. Mereka memiliki potensi, namun perlu strategi khusus untuk mendorong mereka berbelanja.
* **Cluster 2 (Kuning) - Rata-rata:** Pelanggan dengan pendapatan dan skor belanja di tingkat menengah. Mereka adalah basis pelanggan yang stabil.
* **Cluster 3 (Ungu) - Prioritas Rendah:** Pelanggan dengan pendapatan rendah dan skor belanja rendah.
* **Cluster 4 (Merah) - Potensi Terjebak Utang:** Pelanggan dengan pendapatan rendah namun skor belanja tinggi. Kelompok ini perlu diwaspadai agar tidak menjadi pelanggan yang bermasalah.

### Kesimpulan & Rekomendasi Bisnis
Proyek ini berhasil membuktikan bahwa K-Means Clustering adalah alat yang efektif untuk segmentasi pelanggan. Berdasarkan 5 segmen yang ditemukan, tim pemasaran dapat merancang strategi yang ditargetkan:
* **Target Utama:** Berikan program loyalitas eksklusif dan penawaran premium.
* **Hati-hati & Kaya:** Tawarkan produk investasi atau barang mewah yang berkualitas tinggi.
* **Rata-rata:** Libatkan dengan promosi umum dan diskon musiman.

---

### Portofolio 2: Proyek Klasifikasi (Prediksi Gagal Jantung)

Berikut adalah deskripsi profesional untuk proyek klasifikasi Anda.

```markdown
# Model Prediksi Gagal Jantung (Heart Failure) menggunakan Machine Learning

![Heart Failure Banner](https://raw.githubusercontent.com/user-attachments/assets/406793c7-124b-4fd5-ae4e-862d7c504a55)

## Ringkasan Proyek (TL;DR)
Proyek ini bertujuan untuk membangun model *machine learning* yang mampu memprediksi kemungkinan kematian (`DEATH_EVENT`) pada pasien penderita gagal jantung berdasarkan data rekam medis klinis mereka. Proyek ini mencakup penanganan *outliers*, perbandingan beberapa algoritma klasifikasi, dan *hyperparameter tuning* menggunakan `GridSearchCV`. Model **Random Forest Classifier** yang telah dioptimalkan menjadi model terbaik dengan **akurasi mencapai 90%**.

**Tags:** `Python`, `Scikit-Learn`, `Pandas`, `Matplotlib`, `Klasifikasi`, `Machine Learning`, `Kesehatan`, `Analisis Prediktif`

---

## Daftar Isi
1. [Latar Belakang Masalah](#latar-belakang-masalah)
2. [Tujuan Proyek](#tujuan-proyek)
3. [Dataset](#dataset)
4. [Alur Kerja Proyek (Workflow)](#alur-kerja-proyek)
5. [Hasil & Evaluasi](#hasil--evaluasi)
6. [Kesimpulan](#kesimpulan)

---

### Latar Belakang Masalah
Gagal jantung adalah kondisi medis serius dengan tingkat mortalitas yang tinggi. Identifikasi dini pasien yang berisiko tinggi sangat penting untuk memberikan intervensi medis yang tepat waktu dan meningkatkan peluang bertahan hidup. Model prediktif berbasis *machine learning* dapat membantu para profesional medis dalam membuat keputusan klinis dengan menganalisis pola kompleks dari data rekam medis pasien.

### Tujuan Proyek
* Melakukan analisis eksplorasi data untuk memahami hubungan antar fitur klinis.
* Membersihkan data, termasuk menangani nilai pencilan (*outliers*).
* Membangun dan membandingkan performa beberapa model klasifikasi: K-Nearest Neighbors (KNN), Random Forest, dan AdaBoost.
* Mengoptimalkan model terbaik menggunakan `GridSearchCV` untuk menemukan kombinasi *hyperparameter* yang paling akurat.
* Mengevaluasi model akhir menggunakan metrik akurasi dan *confusion matrix*.

### Dataset
Dataset yang digunakan adalah **"Heart Failure Clinical Records"** dari Kaggle. Dataset ini terdiri dari 299 data pasien dengan 13 atribut (fitur), termasuk:
* **Fitur Klinis:** `age`, `anaemia`, `diabetes`, `high_blood_pressure`, `serum_creatinine`, `ejection_fraction`, dll.
* **Target Variable:** `DEATH_EVENT` (1 jika pasien meninggal, 0 jika selamat).

### Alur Kerja Proyek (Workflow)
1.  **Eksplorasi Data (EDA):** Menganalisis korelasi antar fitur menggunakan *heatmap* untuk melihat hubungan linear.
2.  **Penanganan Outliers:** Mengidentifikasi dan menghapus *outliers* menggunakan metode **Interquartile Range (IQR)** untuk memastikan model tidak terdistorsi oleh data ekstrem.
3.  **Persiapan Data:**
    * Membagi dataset menjadi data latih (80%) dan data uji (20%).
    * Melakukan **standardisasi fitur** menggunakan `StandardScaler` agar semua fitur memiliki skala yang sama, yang penting untuk model seperti KNN.
4.  **Pemodelan & Tuning:**
    * Tiga model (KNN, Random Forest, AdaBoost) dilatih pada data latih.
    * Performa awal dievaluasi untuk memilih kandidat model terbaik.
    * **Random Forest** menunjukkan performa awal terbaik, sehingga dipilih untuk optimasi lebih lanjut.
    * **Hyperparameter tuning** dilakukan pada model Random Forest menggunakan `GridSearchCV` untuk mencari parameter optimal seperti `n_estimators`, `max_depth`, dll.

### Hasil & Evaluasi
Proses *tuning* berhasil meningkatkan performa model Random Forest secara signifikan.

* **Perbandingan Akurasi Model (Sebelum & Sesudah Tuning):**

| Model                 | Akurasi Sebelum Tuning | Akurasi Sesudah Tuning |
| --------------------- | ---------------------- | ---------------------- |
| K-Nearest Neighbors   | 83%                    | -                      |
| **Random Forest** | **85%** | **90%** |
| AdaBoost              | 83%                    | -                      |

* **Confusion Matrix (Model Random Forest Tersetel):**
    `Confusion matrix` dari model terbaik pada data uji menunjukkan kemampuan prediksinya:
    * **True Positives & True Negatives:** Model sangat baik dalam mengidentifikasi pasien yang benar-benar selamat dan yang benar-benar meninggal.
    * **False Positives & False Negatives:** Jumlah kesalahan prediksi (misalnya, memprediksi pasien selamat padahal meninggal) relatif rendah.

### Kesimpulan
Proyek ini berhasil mengembangkan model klasifikasi yang andal untuk memprediksi mortalitas akibat gagal jantung dengan **akurasi 90%**. Model Random Forest yang telah dioptimalkan terbukti menjadi alat yang kuat. Model ini berpotensi untuk diintegrasikan sebagai sistem pendukung keputusan bagi para dokter, membantu dalam stratifikasi risiko pasien dan perencanaan perawatan yang lebih proaktif.
