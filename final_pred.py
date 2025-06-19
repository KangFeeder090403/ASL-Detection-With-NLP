# Importing Libraries
import numpy as np
import math
import cv2

import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

offset = 29
hd = HandDetector(maxHands=1)  # Inisialisasi HandDetector untuk deteksi tangan
hd2 = HandDetector(maxHands=1)  # Inisialisasi HandDetector kedua untuk deteksi tambahan

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# Inisialisasi Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()

# Daftar kata dasar bahasa Indonesia dari Sastrawi
KATA_DASAR = set([
    # Kata sapaan dan perkenalan
    'halo', 'hai', 'hi', 'hei', 'kabar', 'nama', 'panggil', 'salam', 'selamat',
    'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam',
    'selamat datang', 'selamat tinggal', 'selamat jalan', 'selamat berjumpa',
    'perkenalkan', 'kenal', 'kenalan', 'teman', 'sahabat', 'kawan', 'rekan',
    'senang', 'senang berkenalan', 'senang bertemu', 'senang berjumpa',
    'terima kasih', 'sama-sama', 'kembali', 'permisi', 'maaf', 'mohon maaf',
    'mohon izin', 'mohon ampun', 'mohon bantuan', 'mohon perhatian',
    
    # Kata ganti
    'saya', 'aku', 'kamu', 'anda', 'dia', 'mereka', 'kita', 'kami',
    'ini', 'itu', 'sana', 'sini', 'situ', 'yang', 'mana', 'apa',
    'siapa', 'berapa', 'bagaimana', 'kapan', 'dimana', 'kenapa',
    
    # Kata penghubung
    'dan', 'atau', 'tetapi', 'karena', 'jika', 'agar', 'supaya', 'sehingga',
    'sebelum', 'sesudah', 'ketika', 'sementara', 'selama', 'sejak', 'hingga',
    'walau', 'meski', 'walaupun', 'meskipun', 'biarpun', 'sekalipun',
    'sebab', 'karena', 'oleh karena', 'dikarenakan', 'lantaran', 'berhubung',
    'maka', 'oleh sebab itu', 'karenanya', 'sehingga', 'maka dari itu',
    'bahwa', 'sehingga', 'agar', 'supaya', 'untuk', 'demi', 'guna',
    'serta', 'dengan', 'bersama', 'sambil', 'seraya', 'selagi', 'selama',
    
    # Kata sifat
    'baik', 'buruk', 'besar', 'kecil', 'tinggi', 'rendah', 'panjang', 'pendek',
    'cepat', 'lambat', 'panas', 'dingin', 'lembut', 'keras', 'halus', 'kasar',
    'cantik', 'jelek', 'muda', 'tua', 'baru', 'lama', 'mahal', 'murah',
    'kuat', 'lemah', 'tebal', 'tipis', 'lebar', 'sempit', 'dalam', 'dangkal',
    'jauh', 'dekat', 'banyak', 'sedikit', 'penuh', 'kosong', 'bersih', 'kotor',
    'rapi', 'berantakan', 'terang', 'gelap', 'terang', 'redup', 'jernih', 'keruh',
    'manis', 'asin', 'pahit', 'pedas', 'asam', 'segar', 'busuk', 'basah',
    'kering', 'lembab', 'licin', 'kesat', 'lunak', 'keras', 'fleksibel', 'kaku',
    'stabil', 'goyah', 'kokoh', 'rapuh', 'kuat', 'lemah', 'tahan', 'rentan',
    'aktif', 'pasif', 'dinamis', 'statis', 'produktif', 'improduktif', 'efisien', 'tidak efisien',
    'hemat', 'boros', 'rajin', 'malas', 'cerdas', 'bodoh', 'pandai', 'tidak pandai',
    'berani', 'penakut', 'jujur', 'curang', 'setia', 'khianat', 'baik hati', 'jahat',
    'ramah', 'kasar', 'sopan', 'tidak sopan', 'sabar', 'tidak sabar', 'tenang', 'gelisah',
    'gembira', 'sedih', 'senang', 'tidak senang', 'bangga', 'malu', 'puas', 'kecewa',
    'bangga', 'rendah hati', 'sombong', 'tidak sombong', 'percaya diri', 'tidak percaya diri',
    'sibuk', 'santai', 'sulit', 'mudah', 'rumit', 'sederhana', 'kompleks', 'dasar',
    'penting', 'sepele', 'utama', 'tambahan', 'khusus', 'umum', 'spesial', 'biasa',
    'unik', 'sama', 'berbeda', 'serupa', 'mirip', 'kontras', 'berlawanan', 'sejalan',
    'sesuai', 'cocok', 'tepat', 'salah', 'benar', 'keliru', 'akurat', 'presisi',
    
    # Kata kerja
    'makan', 'minum', 'tidur', 'bangun', 'jalan', 'lari', 'duduk', 'berdiri',
    'baca', 'tulis', 'dengar', 'lihat', 'ucap', 'pikir', 'rasa', 'tahu',
    'belajar', 'ajar', 'kerja', 'istirahat', 'main', 'bicara', 'ngobrol', 'cerita',
    'ambil', 'beri', 'kirim', 'terima', 'beli', 'jual', 'bayar', 'hitung',
    'masak', 'cuci', 'bersih', 'rapi', 'pakai', 'lepas', 'buka', 'tutup',
    'angkat', 'bawa', 'tarik', 'dorong', 'lempar', 'tangkap', 'pegang', 'sentuh',
    'pukul', 'tendang', 'tusuk', 'potong', 'iris', 'parut', 'giling', 'halus',
    'campur', 'aduk', 'kocok', 'goyang', 'getar', 'putar', 'gulung', 'lipat',
    'ikat', 'jahit', 'rajut', 'tenun', 'anyam', 'pintal', 'belit', 'lilit',
    'sambung', 'hubung', 'pisah', 'bagi', 'gabung', 'satukan', 'lebur', 'cair',
    'beku', 'uap', 'embun', 'tetes', 'alir', 'genang', 'rendam', 'celup',
    'basah', 'kering', 'panas', 'dingin', 'hangat', 'sejuk', 'lembab', 'berkembang',
    'tumbuh', 'berbuah', 'berbunga', 'berdaun', 'berakar', 'berbatang', 'bercabang', 'berkicau',
    'berteriak', 'berbisik', 'bernyanyi', 'berjalan', 'berlari', 'berenang', 'terbang', 'merayap',
    'merangkak', 'berjingkat', 'berjingkrak', 'berputar', 'berputar', 'berputar', 'berputar', 'berputar',
    'cari', 'temu', 'dapat', 'hilang', 'sembunyi', 'tampak', 'muncul', 'lenyap',
    'datang', 'pergi', 'masuk', 'keluar', 'naik', 'turun', 'maju', 'mundur',
    'belok', 'putar', 'balik', 'kembali', 'ulang', 'lanjut', 'berhenti', 'selesai',
    'mulai', 'akhiri', 'buka', 'tutup', 'buka', 'tutup', 'buka', 'tutup',
    'bantu', 'tolong', 'jaga', 'rawat', 'obat', 'sembuh', 'sakit', 'sehat',
    'hidup', 'mati', 'lahir', 'meninggal', 'tua', 'muda', 'besar', 'kecil',
    'tambah', 'kurang', 'kali', 'bagi', 'hitung', 'ukur', 'timbang', 'hitung',
    'pilih', 'ambil', 'beri', 'terima', 'kirim', 'terima', 'beli', 'jual',
    'bayar', 'hutang', 'pinjam', 'kembali', 'simpan', 'ambil', 'pakai', 'lepas',
    'pakai', 'lepas', 'buka', 'tutup', 'buka', 'tutup', 'buka', 'tutup',
    
    # Kata benda
    'rumah', 'sekolah', 'kantor', 'pasar', 'taman', 'jalan', 'kota', 'desa',
    'buku', 'pensil', 'meja', 'kursi', 'telepon', 'komputer', 'laptop', 'hp',
    'mobil', 'motor', 'sepeda', 'pesawat', 'kapal', 'kereta', 'bus', 'truk',
    'pohon', 'bunga', 'daun', 'buah', 'akar', 'batang', 'cabang', 'ranting',
    'hewan', 'burung', 'ikan', 'serangga', 'reptil', 'mamalia', 'unggas', 'binatang',
    'manusia', 'orang', 'anak', 'orang tua', 'kakek', 'nenek', 'paman', 'bibi',
    'saudara', 'teman', 'tetangga', 'guru', 'murid', 'dokter', 'perawat', 'polisi',
    'makanan', 'minuman', 'nasi', 'roti', 'sayur', 'buah', 'daging', 'ikan',
    'air', 'susu', 'teh', 'kopi', 'jus', 'soda', 'sirup', 'minyak',
    'pakaian', 'baju', 'celana', 'rok', 'dasi', 'sepatu', 'sandal', 'topi',
    'perhiasan', 'cincin', 'kalung', 'anting', 'gelang', 'jam', 'kacamata', 'tas',
    'alat', 'pisau', 'garpu', 'sendok', 'piring', 'gelas', 'mangkuk', 'panci',
    'bahan', 'kayu', 'batu', 'besi', 'plastik', 'kaca', 'kain', 'kertas',
    'warna', 'merah', 'biru', 'kuning', 'hijau', 'hitam', 'putih', 'abu-abu',
    'bentuk', 'bulat', 'kotak', 'segi', 'lurus', 'lengkung', 'kerucut', 'piramid',
    'ukuran', 'besar', 'kecil', 'panjang', 'pendek', 'tinggi', 'rendah', 'lebar',
    'sifat', 'keras', 'lembut', 'halus', 'kasar', 'licin', 'kesat', 'basah',
    'kering', 'panas', 'dingin', 'hangat', 'sejuk', 'lembab', 'berat', 'ringan',
    'uang', 'rupiah', 'dolar', 'euro', 'bank', 'atm', 'kartu', 'dompet',
    'waktu', 'jam', 'menit', 'detik', 'hari', 'minggu', 'bulan', 'tahun',
    'musim', 'panas', 'dingin', 'hujan', 'kemarau', 'angin', 'badai', 'topan',
    'alam', 'gunung', 'laut', 'sungai', 'danau', 'hutan', 'padang', 'sawah',
    'udara', 'angin', 'awan', 'hujan', 'salju', 'es', 'embun', 'kabut',
    'cahaya', 'matahari', 'bulan', 'bintang', 'lampu', 'api', 'listrik', 'baterai',
    'suara', 'bunyi', 'musik', 'lagu', 'nyanyi', 'tari', 'tari', 'tari',
    'gambar', 'foto', 'video', 'film', 'televisi', 'radio', 'internet', 'telepon',
    'surat', 'email', 'pesan', 'chat', 'berita', 'koran', 'majalah', 'buku',
    'kata', 'kalimat', 'paragraf', 'cerita', 'puisi', 'lagu', 'syair', 'pantun',
    'bahasa', 'kata', 'kalimat', 'paragraf', 'cerita', 'puisi', 'lagu', 'syair',
    
    # Waktu
    'hari', 'minggu', 'bulan', 'tahun', 'jam', 'menit', 'detik',
    'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu',
    'januari', 'februari', 'maret', 'april', 'mei', 'juni',
    'juli', 'agustus', 'september', 'oktober', 'november', 'desember',
    'pagi', 'siang', 'sore', 'malam', 'dini', 'subuh', 'fajar', 'senja',
    'kemarin', 'hari ini', 'besok', 'lusa', 'sekarang', 'nanti', 'tadi', 'baru',
    'dulu', 'sebelum', 'sesudah', 'selama', 'sementara', 'ketika', 'sejak', 'hingga',
    
    # Angka
    'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh',
    'sebelas', 'dua belas', 'tiga belas', 'empat belas', 'lima belas',
    'enam belas', 'tujuh belas', 'delapan belas', 'sembilan belas', 'dua puluh',
    'dua puluh satu', 'dua puluh dua', 'tiga puluh', 'empat puluh', 'lima puluh',
    'enam puluh', 'tujuh puluh', 'delapan puluh', 'sembilan puluh', 'seratus',
    'seribu', 'sejuta', 'semiliar', 'setriliun',
    'pertama', 'kedua', 'ketiga', 'keempat', 'kelima',
    'kesatu', 'kedua', 'ketiga', 'keempat', 'kelima',
    'sebuah', 'seorang', 'suatu', 'beberapa', 'banyak', 'sedikit',
    'setengah', 'seperempat', 'sepertiga', 'dua pertiga', 'tiga perempat',
    'semua', 'seluruh', 'sebagian', 'sebagian besar', 'sebagian kecil',
    
    # Kata tanya
    'apa', 'siapa', 'kapan', 'dimana', 'kenapa', 'bagaimana', 'berapa',
    'mengapa', 'kenapa', 'bagaimana', 'berapa', 'yang mana', 'dari mana',
    'ke mana', 'untuk apa', 'dengan apa', 'oleh siapa', 'kepada siapa',
    'dari siapa', 'untuk siapa', 'dengan siapa', 'oleh apa', 'kepada apa','bagaimana',
    
    # Kata keterangan
    'sangat', 'sekali', 'terlalu', 'agak', 'sedikit', 'hampir', 'hanya',
    'sudah', 'belum', 'akan', 'sedang', 'pernah', 'selalu', 'kadang',
    'segera', 'cepat', 'lambat', 'pelan', 'perlahan', 'sebentar', 'lama',
    'sekarang', 'nanti', 'besok', 'kemarin', 'lusa', 'tadi', 'baru',
    'di sini', 'di sana', 'di situ', 'di mana', 'ke mana', 'dari mana',
    'ke atas', 'ke bawah', 'ke depan', 'ke belakang', 'ke samping', 'ke tengah',
    'dengan', 'tanpa', 'bersama', 'sendiri', 'berdua', 'bertiga', 'berempat',
    'secara', 'dengan cara', 'dengan baik', 'dengan buruk', 'dengan cepat', 'dengan lambat',
    'sebaiknya', 'seharusnya', 'semestinya', 'sebenarnya', 'sesungguhnya', 'sebetulnya',
    'mungkin', 'barangkali', 'kiranya', 'rasanya', 'sepertinya', 'tampaknya',
    'tentu', 'pasti', 'sudah pasti', 'tidak mungkin', 'mustahil', 'tidak bisa',
    
    # Kata seru
    'ya', 'tidak', 'bukan', 'iya', 'tidak', 'bisa', 'tidak bisa',
    'ah', 'oh', 'wah', 'aduh', 'astaga', 'masyaallah', 'subhanallah',
    'hore', 'hooray', 'wow', 'wow', 'wow', 'wow', 'wow', 'wow',
    'ayo', 'mari', 'silahkan', 'tolong', 'mohon', 'harap', 'minta',
    'maaf', 'permisi', 'terima kasih', 'sama-sama', 'selamat', 'selamat datang',
    
    # Kata depan
    'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'oleh', 'sebagai',
    'antara', 'dalam', 'luar', 'atas', 'bawah', 'depan', 'belakang',
    'samping', 'tengah', 'pinggir', 'ujung', 'pangkal', 'awal', 'akhir',
    'sebelum', 'sesudah', 'selama', 'sementara', 'ketika', 'sejak', 'hingga',
    'karena', 'sebab', 'untuk', 'agar', 'supaya', 'sehingga', 'maka',
    'dengan', 'tanpa', 'bersama', 'sendiri', 'berdua', 'bertiga', 'berempat'
])

# Buat dictionary dari kata dasar
dictionary = ArrayDictionary(KATA_DASAR)

# Daftar kata bahasa Indonesia dengan kata dasar
KATA_BAHASA_INDONESIA = {
    # Kata dasar - Kata Kerja
    'makan', 'minum', 'tidur', 'bangun', 'jalan', 'lari', 'duduk', 'berdiri',
    'baca', 'tulis', 'dengar', 'lihat', 'ucap', 'pikir', 'rasa', 'tahu',
    'ajar', 'kerja', 'main', 'bicara', 'pikir', 'rasa', 'tahu',
    'ambil', 'beri', 'kirim', 'terima', 'beli', 'jual', 'bayar', 'hitung',
    'masak', 'cuci', 'bersih', 'rapi', 'pakai', 'lepas', 'buka', 'tutup',
    'angkat', 'bawa', 'tarik', 'dorong', 'lempar', 'tangkap', 'pegang', 'sentuh',
    'pukul', 'tendang', 'tusuk', 'potong', 'iris', 'parut', 'giling', 'halus',
    'campur', 'aduk', 'kocok', 'goyang', 'getar', 'putar', 'gulung', 'lipat',
    'ikat', 'jahit', 'rajut', 'tenun', 'anyam', 'pintal', 'belit', 'lilit',
    'ikat', 'sambung', 'hubung', 'pisah', 'bagi', 'gabung', 'satukan', 'lebur',
    'cair', 'beku', 'uap', 'embun', 'tetes', 'alir', 'genang', 'rendam',
    'celup', 'basah', 'kering', 'panas', 'dingin', 'hangat', 'sejuk', 'lembab',
    'berkembang', 'tumbuh', 'berbuah', 'berbunga', 'berdaun', 'berakar', 'berbatang', 'bercabang',
    'berkicau', 'berteriak', 'berbisik', 'bernyanyi', 'berteriak', 'berteriak', 'berteriak', 'berteriak',
    'berjalan', 'berlari', 'berenang', 'terbang', 'merayap', 'merangkak', 'berjingkat', 'berjingkrak',
    'berputar', 'berputar', 'berputar', 'berputar', 'berputar', 'berputar', 'berputar', 'berputar',
    'berputar', 'berputar', 'berputar', 'berputar', 'berputar', 'berputar', 'berputar', 'berputar',
    
    # Kata dengan imbuhan - Kata Kerja
    'memakan', 'meminum', 'menidurkan', 'membangunkan', 'berjalan', 'berlari',
    'membaca', 'menulis', 'mendengar', 'melihat', 'mengucap', 'memikirkan',
    'belajar', 'bekerja', 'bermain', 'berbicara', 'merasa', 'mengetahui',
    'mengambil', 'memberikan', 'mengirim', 'menerima', 'membeli', 'menjual',
    'membayar', 'menghitung', 'memasak', 'mencuci', 'membersihkan', 'merapikan',
    'menggunakan', 'melepas', 'membuka', 'menutup', 'mengangkat', 'membawa',
    'menarik', 'mendorong', 'melempar', 'menangkap', 'memegang', 'menyentuh',
    'memukul', 'menendang', 'menusuk', 'memotong', 'mengiris', 'mengparut',
    'menggiling', 'menghaluskan', 'mencampur', 'mengaduk', 'mengocok', 'menggoyang',
    'menggetarkan', 'memutar', 'menggulung', 'melipat', 'mengikat', 'menjahit',
    'merajut', 'menenun', 'menganyam', 'memintal', 'membelit', 'melilit',
    'menyambung', 'menghubung', 'memisah', 'membagi', 'menggabung', 'menyatukan',
    'melebur', 'mencair', 'membeku', 'menguap', 'mengembun', 'menetes',
    'mengalir', 'menggenang', 'merendam', 'mencelup', 'membasah', 'mengering',
    'memanas', 'mendingin', 'menghangat', 'menyejuk', 'melembab',
    
    # Kata dasar - Kata Benda
    'rumah', 'sekolah', 'kantor', 'pasar', 'taman', 'jalan', 'kota', 'desa',
    'buku', 'pensil', 'meja', 'kursi', 'telepon', 'komputer', 'laptop',
    'mobil', 'motor', 'sepeda', 'pesawat', 'kapal', 'kereta',
    'pohon', 'bunga', 'daun', 'buah', 'akar', 'batang', 'cabang', 'ranting',
    'hewan', 'burung', 'ikan', 'serangga', 'reptil', 'mamalia', 'unggas', 'binatang',
    'manusia', 'orang', 'anak', 'orang tua', 'kakek', 'nenek', 'paman', 'bibi',
    'saudara', 'teman', 'tetangga', 'guru', 'murid', 'dokter', 'perawat', 'polisi',
    'makanan', 'minuman', 'nasi', 'roti', 'sayur', 'buah', 'daging', 'ikan',
    'air', 'susu', 'teh', 'kopi', 'jus', 'soda', 'sirup', 'minyak',
    'pakaian', 'baju', 'celana', 'rok', 'dasi', 'sepatu', 'sandal', 'topi',
    'perhiasan', 'cincin', 'kalung', 'anting', 'gelang', 'jam', 'kacamata', 'tas',
    'alat', 'pisau', 'garpu', 'sendok', 'piring', 'gelas', 'mangkuk', 'panci',
    'bahan', 'kayu', 'batu', 'besi', 'plastik', 'kaca', 'kain', 'kertas',
    'warna', 'merah', 'biru', 'kuning', 'hijau', 'hitam', 'putih', 'abu-abu',
    'bentuk', 'bulat', 'kotak', 'segi', 'lurus', 'lengkung', 'kerucut', 'piramid',
    'ukuran', 'besar', 'kecil', 'panjang', 'pendek', 'tinggi', 'rendah', 'lebar',
    'sifat', 'keras', 'lembut', 'halus', 'kasar', 'licin', 'kesat', 'basah',
    'kering', 'panas', 'dingin', 'hangat', 'sejuk', 'lembab', 'berat', 'ringan',
    
    # Kata dasar - Kata Sifat
    'baik', 'buruk', 'besar', 'kecil', 'tinggi', 'rendah', 'panjang', 'pendek',
    'cepat', 'lambat', 'panas', 'dingin', 'lembut', 'keras', 'halus', 'kasar',
    'cantik', 'jelek', 'muda', 'tua', 'baru', 'lama', 'mahal', 'murah',
    'kuat', 'lemah', 'tebal', 'tipis', 'lebar', 'sempit', 'dalam', 'dangkal',
    'jauh', 'dekat', 'banyak', 'sedikit', 'penuh', 'kosong', 'bersih', 'kotor',
    'rapi', 'berantakan', 'terang', 'gelap', 'terang', 'redup', 'jernih', 'keruh',
    'manis', 'asin', 'pahit', 'pedas', 'asam', 'segar', 'busuk', 'basah',
    'kering', 'lembab', 'licin', 'kesat', 'lunak', 'keras', 'fleksibel', 'kaku',
    'stabil', 'goyah', 'kokoh', 'rapuh', 'kuat', 'lemah', 'tahan', 'rentan',
    'aktif', 'pasif', 'dinamis', 'statis', 'produktif', 'improduktif', 'efisien', 'tidak efisien',
    'hemat', 'boros', 'rajin', 'malas', 'cerdas', 'bodoh', 'pandai', 'tidak pandai',
    'berani', 'penakut', 'jujur', 'curang', 'setia', 'khianat', 'baik hati', 'jahat',
    'ramah', 'kasar', 'sopan', 'tidak sopan', 'sabar', 'tidak sabar', 'tenang', 'gelisah',
    'gembira', 'sedih', 'senang', 'tidak senang', 'bangga', 'malu', 'puas', 'kecewa',
    'bangga', 'rendah hati', 'sombong', 'tidak sombong', 'percaya diri', 'tidak percaya diri',
    
    # Kata dasar - Kata Keterangan
    'sangat', 'sekali', 'terlalu', 'agak', 'sedikit', 'hampir', 'hanya',
    'sudah', 'belum', 'akan', 'sedang', 'pernah', 'selalu', 'kadang',
    'segera', 'cepat', 'lambat', 'pelan', 'perlahan', 'sebentar', 'lama',
    'sekarang', 'nanti', 'besok', 'kemarin', 'lusa', 'tadi', 'baru',
    'di sini', 'di sana', 'di situ', 'di mana', 'ke mana', 'dari mana',
    'ke atas', 'ke bawah', 'ke depan', 'ke belakang', 'ke samping', 'ke tengah',
    'dengan', 'tanpa', 'bersama', 'sendiri', 'berdua', 'bertiga', 'berempat',
    'secara', 'dengan cara', 'dengan baik', 'dengan buruk', 'dengan cepat', 'dengan lambat',
    'sebaiknya', 'seharusnya', 'semestinya', 'sebenarnya', 'sesungguhnya', 'sebetulnya',
    'mungkin', 'barangkali', 'kiranya', 'rasanya', 'sepertinya', 'tampaknya',
    'tentu', 'pasti', 'sudah pasti', 'tidak mungkin', 'mustahil', 'tidak bisa',
    
    # Kata dasar - Kata Bilangan
    'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh',
    'sebelas', 'dua belas', 'tiga belas', 'empat belas', 'lima belas',
    'enam belas', 'tujuh belas', 'delapan belas', 'sembilan belas', 'dua puluh',
    'dua puluh satu', 'dua puluh dua', 'tiga puluh', 'empat puluh', 'lima puluh',
    'enam puluh', 'tujuh puluh', 'delapan puluh', 'sembilan puluh', 'seratus',
    'seribu', 'sejuta', 'semiliar', 'setriliun',
    'pertama', 'kedua', 'ketiga', 'keempat', 'kelima',
    'kesatu', 'kedua', 'ketiga', 'keempat', 'kelima',
    'sebuah', 'seorang', 'suatu', 'beberapa', 'banyak', 'sedikit',
    'setengah', 'seperempat', 'sepertiga', 'dua pertiga', 'tiga perempat',
    'semua', 'seluruh', 'sebagian', 'sebagian besar', 'sebagian kecil',
    
    # Kata dasar - Kata Tanya
    'apa', 'siapa', 'kapan', 'dimana', 'kenapa', 'bagaimana', 'berapa',
    'mengapa', 'kenapa', 'bagaimana', 'berapa', 'yang mana', 'dari mana',
    'ke mana', 'untuk apa', 'dengan apa', 'oleh siapa', 'kepada siapa',
    'dari siapa', 'untuk siapa', 'dengan siapa', 'oleh apa', 'kepada apa',
    
    # Kata dasar - Kata Penghubung
    'dan', 'atau', 'tetapi', 'karena', 'jika', 'agar', 'supaya', 'sehingga',
    'sebelum', 'sesudah', 'ketika', 'sementara', 'selama', 'sejak', 'hingga',
    'walau', 'meski', 'walaupun', 'meskipun', 'biarpun', 'sekalipun',
    'sebab', 'karena', 'oleh karena', 'dikarenakan', 'lantaran', 'berhubung',
    'maka', 'oleh sebab itu', 'karenanya', 'sehingga', 'maka dari itu',
    'bahwa', 'sehingga', 'agar', 'supaya', 'untuk', 'demi', 'guna',
    'serta', 'dengan', 'bersama', 'sambil', 'seraya', 'selagi', 'selama',
    
    # Kata dasar - Kata Seru
    'ya', 'tidak', 'bukan', 'iya', 'tidak', 'bisa', 'tidak bisa',
    'ah', 'oh', 'wah', 'aduh', 'astaga', 'masyaallah', 'subhanallah',
    'hore', 'hooray', 'wow', 'wow', 'wow', 'wow', 'wow', 'wow',
    'ayo', 'mari', 'silahkan', 'tolong', 'mohon', 'harap', 'minta',
    'maaf', 'permisi', 'terima kasih', 'sama-sama', 'selamat', 'selamat datang',
    
    # Kata dasar - Kata Depan
    'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'oleh', 'sebagai',
    'antara', 'dalam', 'luar', 'atas', 'bawah', 'depan', 'belakang',
    'samping', 'tengah', 'pinggir', 'ujung', 'pangkal', 'awal', 'akhir',
    'sebelum', 'sesudah', 'selama', 'sementara', 'ketika', 'sejak', 'hingga',
    'karena', 'sebab', 'untuk', 'agar', 'supaya', 'sehingga', 'maka',
    'dengan', 'tanpa', 'bersama', 'sendiri', 'berdua', 'bertiga', 'berempat'
}

# Application :
class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')
        self.speak_engine=pyttsx3.init()
        self.speak_engine.setProperty("rate",100)
        voices=self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice",voices[0].id)

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.space_flag=False
        self.next_flag=True
        self.prev_char=""
        self.count=-1
        self.ten_prev_char=[]
        for i in range(10):
            self.ten_prev_char.append(" ")


        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")

        # --- Modern GUI Layout ---
        self.root = tk.Tk()
        self.root.title("Penerjemah Bahasa Isyarat ke Teks")
        self.root.geometry("1200x750")
        self.root.configure(bg='#f4f6fb')
        self.root.resizable(True, True)

        # Set up grid weights for responsiveness
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Title
        self.T = tk.Label(self.root, text="Penerjemah Bahasa Isyarat ke Teks", font=("Segoe UI", 32, "bold"), bg='#f4f6fb', fg='#2d3436')
        self.T.grid(row=0, column=0, columnspan=3, pady=(20, 10), sticky="nsew")

        # Video and Detection Panels
        self.panel = tk.Label(self.root, bg='#fff', relief='solid', borderwidth=2, width=50, height=20)
        self.panel.grid(row=1, column=0, padx=(40, 20), pady=10, sticky="nsew")

        self.panel2 = tk.Label(self.root, bg='#fff', relief='solid', borderwidth=2, width=40, height=20)
        self.panel2.grid(row=1, column=1, padx=(20, 40), pady=10, sticky="nsew")

        # Detected Character and Current Text
        info_frame = tk.Frame(self.root, bg='#f4f6fb')
        info_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        info_frame.grid_columnconfigure(1, weight=1)
        info_frame.grid_columnconfigure(3, weight=2)

        self.T1 = tk.Label(info_frame, text="Karakter Terdeteksi:", font=("Segoe UI", 16, "bold"), bg='#f4f6fb', fg='#2d3436')
        self.T1.grid(row=0, column=0, padx=(10, 5), sticky="e")
        self.panel3 = tk.Label(info_frame, font=("Segoe UI", 18, "bold"), bg='#fff', relief='solid', borderwidth=1, width=10)
        self.panel3.grid(row=0, column=1, padx=(0, 30), sticky="w")

        self.T3 = tk.Label(info_frame, text="Teks Saat Ini:", font=("Segoe UI", 16, "bold"), bg='#f4f6fb', fg='#2d3436')
        self.T3.grid(row=0, column=2, padx=(10, 5), sticky="e")
        self.panel5 = tk.Label(info_frame, font=("Segoe UI", 16), bg='#fff', relief='solid', borderwidth=1, width=40, anchor='w')
        self.panel5.grid(row=0, column=3, padx=(0, 10), sticky="ew")

        # Suggestions
        self.T4 = tk.Label(self.root, text="Saran Kata:", font=("Segoe UI", 20, "bold"), bg='#f4f6fb', fg='#e17055')
        self.T4.grid(row=3, column=0, columnspan=2, pady=(30, 5), sticky="w", padx=(60,0))

        sugg_frame = tk.Frame(self.root, bg='#f4f6fb')
        sugg_frame.grid(row=4, column=0, columnspan=2, pady=(0, 20), sticky="ew")
        for i in range(4):
            sugg_frame.grid_columnconfigure(i, weight=1)

        button_style = {
            'font': ('Segoe UI', 14, 'bold'),
            'width': 15,
            'height': 2,
            'bg': '#0984e3',
            'fg': 'white',
            'relief': 'flat',
            'activebackground': '#74b9ff',
            'activeforeground': 'white',
            'cursor': 'hand2',
            'wraplength': 200,  # Memungkinkan text wrapping
            'justify': 'left'   # Text alignment
        }
        self.b1 = tk.Button(sugg_frame, command=self.action1, **button_style)
        self.b1.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.b2 = tk.Button(sugg_frame, command=self.action2, **button_style)
        self.b2.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.b3 = tk.Button(sugg_frame, command=self.action3, **button_style)
        self.b3.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        self.b4 = tk.Button(sugg_frame, command=self.action4, **button_style)
        self.b4.grid(row=0, column=3, padx=10, pady=5, sticky="ew")

        # Control Buttons (Speak & Clear)
        ctrl_frame = tk.Frame(self.root, bg='#f4f6fb')
        ctrl_frame.grid(row=5, column=0, columnspan=2, pady=(10, 10))
        ctrl_frame.grid_columnconfigure(0, weight=1)
        ctrl_frame.grid_columnconfigure(1, weight=1)

        control_button_style = {
            'font': ('Segoe UI', 14, 'bold'),
            'width': 12,
            'height': 1,
            'bg': '#00b894',
            'fg': 'white',
            'relief': 'flat',
            'activebackground': '#55efc4',
            'activeforeground': 'white',
            'cursor': 'hand2',
        }
        self.speak = tk.Button(ctrl_frame, text="🔊 Baca Teks", command=self.speak_fun, **control_button_style)
        self.speak.grid(row=0, column=0, padx=20, pady=5, sticky="e")
        self.clear = tk.Button(ctrl_frame, text="Clear", command=self.clear_fun, **control_button_style)
        self.clear.grid(row=0, column=1, padx=20, pady=5, sticky="w")

        # Initialize variables
        self.str = " "
        self.ccc = 0
        self.word = " "
        self.current_symbol = "C"
        self.photo = "Empty"
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        self.word_history = deque(maxlen=5)  # Menyimpan 5 kata terakhir untuk konteks
        self.current_context = []  # Kata-kata yang mungkin relevan berdasarkan konteks

        # Tambahkan label untuk konteks
        self.context_label = tk.Label(self.root, text="Konteks:", font=("Segoe UI", 12), bg='#f4f6fb', fg='#2d3436')
        self.context_label.grid(row=5, column=0, columnspan=2, pady=(10,0), sticky="w", padx=(60,0))
        
        self.context_text = tk.Label(self.root, text="", font=("Segoe UI", 12), bg='#f4f6fb', fg='#636e72', wraplength=1000)
        self.context_text.grid(row=6, column=0, columnspan=2, pady=(0,10), sticky="w", padx=(60,0))

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            if cv2image.any:
                hands = hd.findHands(cv2image, draw=False, flipType=True)
                cv2image_copy=np.array(cv2image)
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)

                if hands[0]:
                    hand = hands[0]
                    map = hand[0]
                    x, y, w, h=map['bbox']
                    image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                    white = cv2.imread("white.jpg")
                    # img_final=img_final1=img_final2=0
                    if image.all:
                        handz = hd2.findHands(image, draw=False, flipType=True)
                        self.ccc += 1
                        if handz[0]:
                            hand = handz[0]
                            handmap=hand[0]
                            self.pts = handmap['lmList']
                            # x1,y1,w1,h1=hand['bbox']

                            os = ((400 - w) // 2) - 15
                            os1 = ((400 - h) // 2) - 15
                            for t in range(0, 4, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(5, 8, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(9, 12, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(13, 16, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            for t in range(17, 20, 1):
                                cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                         (0, 255, 0), 3)
                            cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                                     (0, 255, 0), 3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                                     3)
                            cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                                     3)

                            for i in range(21):
                                cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                            res=white
                            self.predict(res)

                            self.current_image2 = Image.fromarray(res)

                            imgtk = ImageTk.PhotoImage(image=self.current_image2)

                            self.panel2.imgtk = imgtk
                            self.panel2.config(image=imgtk)

                            self.panel3.config(text=self.current_symbol, font=("Segoe UI", 18, "bold"))

                            #self.panel4.config(text=self.word, font=("Courier", 30))



                            self.b1.config(text=self.word1, font=("Segoe UI", 14, "bold"), wraplength=825, command=self.action1)
                            self.b2.config(text=self.word2, font=("Segoe UI", 14, "bold"), wraplength=825,  command=self.action2)
                            self.b3.config(text=self.word3, font=("Segoe UI", 14, "bold"), wraplength=825,  command=self.action3)
                            self.b4.config(text=self.word4, font=("Segoe UI", 14, "bold"), wraplength=825,  command=self.action4)

                self.panel5.config(text=self.str, font=("Segoe UI", 16), wraplength=1025)
        except Exception:
            print(Exception.__traceback__)
            hands = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy=np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                # #print(" --------- lmlist=",hands[1])
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                white = cv2.imread("d:\\Semster 6\\Pengolahaan Bahasa Alamai\\tubes4\\Asl hore\\white.jpg")
                # img_final=img_final1=img_final2=0
                

                handz = hd2.findHands(image, draw=False, flipType=True)
                print(" ", self.ccc)
                self.ccc += 1
                if handz:
                    hand = handz[0]
                    self.pts = hand['lmList']
                    # x1,y1,w1,h1=hand['bbox']

                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    for t in range(0, 4, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (self.pts[t][0] + os, self.pts[t][1] + os1), (self.pts[t + 1][0] + os, self.pts[t + 1][1] + os1),
                                 (0, 255, 0), 3)
                    cv2.line(white, (self.pts[5][0] + os, self.pts[5][1] + os1), (self.pts[9][0] + os, self.pts[9][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[9][0] + os, self.pts[9][1] + os1), (self.pts[13][0] + os, self.pts[13][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[13][0] + os, self.pts[13][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1),
                             (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[5][0] + os, self.pts[5][1] + os1), (0, 255, 0),
                             3)
                    cv2.line(white, (self.pts[0][0] + os, self.pts[0][1] + os1), (self.pts[17][0] + os, self.pts[17][1] + os1), (0, 255, 0),
                             3)

                    for i in range(21):
                        cv2.circle(white, (self.pts[i][0] + os, self.pts[i][1] + os1), 2, (0, 0, 255), 1)

                    res=white
                    self.predict(res)

                    self.current_image2 = Image.fromarray(res)

                    imgtk = ImageTk.PhotoImage(image=self.current_image2)

                    self.panel2.imgtk = imgtk
                    self.panel2.config(image=imgtk)

                    self.panel3.config(text=self.current_symbol, font=("Segoe UI", 18, "bold"))

                    #self.panel4.config(text=self.word, font=("Courier", 30))



                    self.b1.config(text=self.word1, font=("Segoe UI", 14, "bold"), wraplength=825, command=self.action1)
                    self.b2.config(text=self.word2, font=("Segoe UI", 14, "bold"), wraplength=825,  command=self.action2)
                    self.b3.config(text=self.word3, font=("Segoe UI", 14, "bold"), wraplength=825,  command=self.action3)
                    self.b4.config(text=self.word4, font=("Segoe UI", 14, "bold"), wraplength=825,  command=self.action4)

            self.panel5.config(text=self.str, font=("Segoe UI", 16), wraplength=1025)
        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def get_word_suggestions(self, word, max_suggestions=4):
        """Mendapatkan saran kata menggunakan Sastrawi"""
        if not word.strip():
            return []
            
        word = word.lower()
        suggestions = []
        
        # Coba stem kata untuk mendapatkan kata dasar
        stemmed_word = stemmer.stem(word)
        
        # Cari kata yang mirip berdasarkan awalan dan kata dasar
        for dict_word in KATA_BAHASA_INDONESIA:
            dict_word = dict_word.lower()
            # Cek awalan
            if dict_word.startswith(word):
                suggestions.append(dict_word)
            # Cek kata dasar
            elif stemmer.stem(dict_word) == stemmed_word and dict_word != word:
                suggestions.append(dict_word)
                
        # Tambahkan konteks dari kata sebelumnya
        if self.word_history:
            for dict_word in KATA_BAHASA_INDONESIA:
                dict_word = dict_word.lower()
                # Cek apakah kata ini sering muncul setelah kata terakhir
                if any(dict_word in hist.lower() for hist in self.word_history):
                    suggestions.append(dict_word)
                    
        # Urutkan berdasarkan:
        # 1. Dimulai dengan kata yang sama
        # 2. Memiliki kata dasar yang sama
        # 3. Pernah digunakan dalam konteks
        # 4. Panjang kata
        suggestions = list(set(suggestions))  # Hapus duplikat
        suggestions.sort(key=lambda x: (
            x.startswith(word),  # Prioritaskan yang dimulai sama
            stemmer.stem(x) == stemmed_word,  # Kemudian yang memiliki kata dasar sama
            x in [w.lower() for w in self.word_history],  # Kemudian yang pernah digunakan
            -len(x)  # Terakhir berdasarkan panjang
        ), reverse=True)
        
        return suggestions[:max_suggestions]

    def update_suggestions(self, word):
        """Update suggestion menggunakan Sastrawi"""
        if not word.strip():
            self.word1 = self.word2 = self.word3 = self.word4 = " "
            self.b1.config(text=" ")
            self.b2.config(text=" ")
            self.b3.config(text=" ")
            self.b4.config(text=" ")
            return
            
        # Dapatkan saran kata
        suggestions = self.get_word_suggestions(word)
        
        # Update suggestion buttons
        self.word1 = suggestions[0] if len(suggestions) > 0 else " "
        self.word2 = suggestions[1] if len(suggestions) > 1 else " "
        self.word3 = suggestions[2] if len(suggestions) > 2 else " "
        self.word4 = suggestions[3] if len(suggestions) > 3 else " "
        
        self.b1.config(text=self.word1)
        self.b2.config(text=self.word2)
        self.b3.config(text=self.word3)
        self.b4.config(text=self.word4)
        
        # Update konteks
        if self.word_history:
            context_words = [w for w in self.word_history if w.strip()]
            if context_words:
                # Tampilkan kata dasar dari kata-kata sebelumnya
                stemmed_context = [stemmer.stem(w) for w in context_words[-3:]]
                self.context_text.config(text="Kata sebelumnya: " + " → ".join(context_words[-3:]) + 
                                      "\nKata dasar: " + " → ".join(stemmed_context))

    def action1(self):
        if self.word1.strip():
            idx_space = self.str.rfind(" ")
            idx_word = self.str.find(self.word, idx_space)
            self.str = self.str[:idx_word] + self.word1
            self.word_history.append(self.word1)
            self.update_suggestions("")  # Reset suggestions

    def action2(self):
        if self.word2.strip():
            idx_space = self.str.rfind(" ")
            idx_word = self.str.find(self.word, idx_space)
            self.str = self.str[:idx_word] + self.word2
            self.word_history.append(self.word2)
            self.update_suggestions("")

    def action3(self):
        if self.word3.strip():
            idx_space = self.str.rfind(" ")
            idx_word = self.str.find(self.word, idx_space)
            self.str = self.str[:idx_word] + self.word3
            self.word_history.append(self.word3)
            self.update_suggestions("")

    def action4(self):
        if self.word4.strip():
            idx_space = self.str.rfind(" ")
            idx_word = self.str.find(self.word, idx_space)
            self.str = self.str[:idx_word] + self.word4
            self.word_history.append(self.word4)
            self.update_suggestions("")

    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()


    def clear_fun(self):
        self.str = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.word_history.clear()
        self.update_suggestions("")
        self.context_text.config(text="")

    def predict(self, test_image):
        white=test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1=" "



        print(self.pts[4][0] < self.pts[5][0])
        if ch1 == 'E' or ch1=='Y' or ch1=='B':
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1="next"


        if ch1 == 'Next' or 'B' or 'C' or 'H' or 'F' or 'X':
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]):
                ch1 = 'Backspace'


        if ch1=="next" and self.prev_char!="next":
            if self.ten_prev_char[(self.count-2)%10]!="next":
                if self.ten_prev_char[(self.count-2)%10]=="Backspace":
                    self.str=self.str[0:-1]
                else:
                    if self.ten_prev_char[(self.count - 2) % 10] != "Backspace":
                        self.str = self.str + self.ten_prev_char[(self.count-2)%10]
            else:
                if self.ten_prev_char[(self.count - 0) % 10] != "Backspace":
                    self.str = self.str + self.ten_prev_char[(self.count - 0) % 10]


        if ch1=="  " and self.prev_char!="  ":
            self.str = self.str + "  "

        self.prev_char=ch1
        self.current_symbol=ch1
        self.count += 1
        self.ten_prev_char[self.count%10]=ch1


        if len(self.str.strip())!=0:
            st=self.str.rfind(" ")
            ed=len(self.str)
            word=self.str[st+1:ed]
            self.word=word
            if len(word.strip())!=0:
                self.update_suggestions(word)
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = " "
                self.b1.config(text=" ")
                self.b2.config(text=" ")
                self.b3.config(text=" ")
                self.b4.config(text=" ")


    def destructor(self):
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")

(Application()).root.mainloop()
