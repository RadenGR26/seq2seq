import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)


# --- Fungsi untuk memuat data dari file teks ---
def load_sentences_from_file(filepath):
    """
    Memuat kalimat dari file teks, setiap baris adalah satu kalimat.
    """
    sentences = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                sentences.append(line.strip())  # Menghapus karakter newline
    except FileNotFoundError:
        print(f"File tidak ditemukan: {filepath}")
    except Exception as e:
        print(f"Terjadi kesalahan saat membaca file {filepath}: {e}")
    return sentences


# --- Direktori Data ---
DATA_DIR = 'data'
ENGLISH_FILE = os.path.join(DATA_DIR, 'english_sentences.txt')
FRENCH_FILE = os.path.join(DATA_DIR, 'french_sentences.txt')
INDONESIAN_FILE = os.path.join(DATA_DIR, 'indonesian_sentences.txt')

# Variabel global untuk menyimpan data dan model
english_sentences_loaded = []
french_sentences_loaded = []
indonesian_sentences_loaded = []
eng_word2index = {}
fre_word2index = {}
ind_word2index = {}
eng_index2word = {}
fre_index2word = {}
ind_index2word = {}
model = None


# --- Inisialisasi Data dan Model saat aplikasi dimulai ---
def initialize_model_and_data():
    global english_sentences_loaded, french_sentences_loaded, indonesian_sentences_loaded, \
        eng_word2index, fre_word2index, ind_word2index, \
        eng_index2word, fre_index2word, ind_index2word, model

    try:
        english_sentences_loaded = load_sentences_from_file(ENGLISH_FILE)
        french_sentences_loaded = load_sentences_from_file(FRENCH_FILE)
        indonesian_sentences_loaded = load_sentences_from_file(INDONESIAN_FILE)

        if not english_sentences_loaded or not french_sentences_loaded or not indonesian_sentences_loaded:
            raise ValueError("File dataset kosong atau tidak dapat dimuat sepenuhnya.")

        if not (len(english_sentences_loaded) == len(french_sentences_loaded) == len(indonesian_sentences_loaded)):
            raise ValueError("Jumlah kalimat dalam file bahasa tidak sama.")

        print(f"Berhasil memuat {len(indonesian_sentences_loaded)} pasangan kalimat untuk 3 bahasa.")

        # Tokenization & Vocabulary Building
        all_english_words = ' '.join(english_sentences_loaded).split()
        all_french_words = ' '.join(french_sentences_loaded).split()
        all_indonesian_words = ' '.join(indonesian_sentences_loaded).split()

        english_vocab_set = sorted(list(set(all_english_words)))
        french_vocab_set = sorted(list(set(all_french_words)))
        indonesian_vocab_set = sorted(list(set(all_indonesian_words)))

        # Tambahkan token khusus.
        SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

        # Buat word-to-index dan index-to-word dictionaries
        for i, word in enumerate(SPECIAL_TOKENS + english_vocab_set):
            eng_word2index[word] = i
            eng_index2word[i] = word

        for i, word in enumerate(SPECIAL_TOKENS + french_vocab_set):
            fre_word2index[word] = i
            fre_index2word[i] = word

        for i, word in enumerate(SPECIAL_TOKENS + indonesian_vocab_set):
            ind_word2index[word] = i
            ind_index2word[i] = word

        # --- Seq2Seq Class Definition (Disesuaikan untuk Input Indonesia) ---
        class Seq2Seq(nn.Module):
            def __init__(self, input_size_id, output_size_eng, output_size_fr, hidden_size):
                super(Seq2Seq, self).__init__()
                self.hidden_size = hidden_size
                # Encoder untuk bahasa Indonesia
                self.encoder = nn.Embedding(input_size_id, hidden_size)  # Input sekarang Bahasa Indonesia
                self.rnn_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)

                # Decoders terpisah untuk setiap bahasa target
                self.decoder_eng = nn.Embedding(output_size_eng, hidden_size)  # Output Bahasa Inggris
                self.rnn_decoder_eng = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                self.fc_eng = nn.Linear(hidden_size, output_size_eng)

                self.decoder_fr = nn.Embedding(output_size_fr, hidden_size)  # Output Bahasa Prancis
                self.rnn_decoder_fr = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                self.fc_fr = nn.Linear(hidden_size, output_size_fr)

            def forward(self, input_tensor_id, target_tensor_eng, target_tensor_fr):
                # Encoder (mengambil input Bahasa Indonesia)
                embedded_input_id = self.encoder(input_tensor_id)
                _, (hidden, cell) = self.rnn_encoder(embedded_input_id)

                # Decoder Bahasa Inggris
                embedded_target_eng = self.decoder_eng(target_tensor_eng)
                output_eng, _ = self.rnn_decoder_eng(embedded_target_eng, (hidden, cell))
                output_eng = self.fc_eng(output_eng)

                # Decoder Bahasa Prancis
                embedded_target_fr = self.decoder_fr(target_tensor_fr)
                output_fr, _ = self.rnn_decoder_fr(embedded_target_fr, (hidden, cell))
                output_fr = self.fc_fr(output_fr)

                return output_eng, output_fr  # Mengembalikan output Bahasa Inggris dan Prancis

        # --- Inisialisasi Model ---
        input_size_id = len(ind_word2index)
        output_size_eng = len(eng_word2index)
        output_size_fr = len(fre_word2index)
        hidden_size = 256
        model = Seq2Seq(input_size_id, output_size_eng, output_size_fr, hidden_size)

        # Inisialisasi loss function dan optimizer (tanpa pelatihan nyata di sini)
        criterion_eng = nn.CrossEntropyLoss(ignore_index=eng_word2index.get('<PAD>', -1))
        criterion_fr = nn.CrossEntropyLoss(ignore_index=fre_word2index.get('<PAD>', -1))
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print("Model Seq2Seq untuk terjemahan Indonesia ke Inggris/Prancis berhasil diinisialisasi.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Gagal memuat dataset atau menginisialisasi model: {e}")
        model = None
        english_sentences_loaded = []
        french_sentences_loaded = []
        indonesian_sentences_loaded = []
    except Exception as e:
        print(f"Terjadi kesalahan tak terduga saat inisialisasi: {e}")
        model = None
        english_sentences_loaded = []
        french_sentences_loaded = []
        indonesian_sentences_loaded = []


# Panggil inisialisasi saat aplikasi dimulai
initialize_model_and_data()


# --- Fungsi bantuan untuk menerjemahkan kalimat (placeholder untuk model yang belum dilatih) ---
def translate_sentence(sentence_to_translate, current_model):
    """
    Fungsi placeholder untuk terjemahan dari Bahasa Indonesia ke Inggris dan Prancis.
    Dalam aplikasi nyata, ini akan memanggil inferensi model Seq2Seq yang sudah dilatih.
    Saat ini, hanya akan "menerjemahkan" jika kalimat ada di dataset yang dimuat.
    Mengembalikan dictionary dengan terjemahan Inggris dan Prancis.
    """
    if current_model is None:
        return {
            "english": "Sistem terjemahan tidak aktif.",
            "french": "Sistem terjemahan tidak aktif."
        }

    normalized_sentence = sentence_to_translate.strip()

    # Cari di daftar kalimat Bahasa Indonesia yang dimuat
    if normalized_sentence in indonesian_sentences_loaded:
        idx = indonesian_sentences_loaded.index(normalized_sentence)
        return {
            "english": english_sentences_loaded[idx],  # Ambil terjemahan Bahasa Inggris
            "french": french_sentences_loaded[idx]  # Ambil terjemahan Bahasa Prancis
        }
    else:
        # Jika kalimat tidak ada di dataset, tunjukkan bahwa model perlu dilatih.
        return {
            "english": "Terjemahan tidak tersedia (model belum dilatih).",
            "french": "Terjemahan tidak tersedia (model belum dilatih)."
        }


# --- Flask Routes ---
@app.route('/')
def index():
    # Siapkan contoh-contoh untuk ditampilkan di halaman HTML
    # Ambil hingga 10 contoh pertama dari dataset yang dimuat
    display_examples = []
    for i in range(min(10, len(indonesian_sentences_loaded))):
        display_examples.append({
            "indonesian_input": indonesian_sentences_loaded[i],  # Inputnya sekarang Indonesia
            "english_output": english_sentences_loaded[i],  # Output Bahasa Inggris
            "french_output": french_sentences_loaded[i]  # Output Bahasa Prancis
        })
    return render_template('index.html', examples=display_examples)


@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    indonesian_text = data.get('text', '')  # Input sekarang dari Bahasa Indonesia

    if not indonesian_text:
        return jsonify({'error': 'No text provided'}), 400

    # Lakukan terjemahan menggunakan fungsi placeholder
    translated_texts = translate_sentence(indonesian_text, model)

    # Mengembalikan kedua terjemahan
    return jsonify({
        'indonesian_input': indonesian_text,
        'english': translated_texts['english'],  # Key 'english' untuk hasil terjemahan Inggris
        'french': translated_texts['french']  # Key 'french' untuk hasil terjemahan Prancis
    })


if __name__ == '__main__':
    print("Aplikasi Flask akan dimulai. Kunjungi http://127.0.0.1:5000/")
    app.run(debug=True)