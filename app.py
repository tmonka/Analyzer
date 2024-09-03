from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import spacy
from string import punctuation
from heapq import nlargest
from PyPDF2 import PdfReader
from models import db, User  # Импортируем db и User

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# Определение путей к директориям uploads и summaries
base_dir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads/')
app.config['SUMMARY_FOLDER'] = os.path.join(base_dir, 'summaries/')

db.init_app(app)  # Инициализация SQLAlchemy с текущим приложением

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Загрузка русской модели SpaCy
nlp = spacy.load('ru_core_news_sm')

@login_manager.user_loader
def load_user(user_id):
    with app.app_context():
        session = db.session
        return session.get(User, int(user_id))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Имя пользователя уже существует!')
            return redirect(url_for('register'))

        new_user = User(username=username, password=generate_password_hash(password, method='sha256'))
        db.session.add(new_user)
        db.session.commit()

        flash('Регистрация прошла успешно! Пожалуйста, войдите в систему.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('upload'))

        flash('Неверное имя пользователя или пароль.')
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'document' not in request.files:
            flash('Нет файла для загрузки')
            return redirect(request.url)

        file = request.files['document']
        if file.filename == '':
            flash('Файл не выбран')
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"Файл сохранен: {file_path}")  # Логирование
            flash('Файл успешно загружен.')
            return redirect(url_for('list_files'))

    return render_template('upload.html')

@app.route('/files')
@login_required
def list_files():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        print(f"Загруженные файлы: {files}")  # Логирование списка файлов
        return render_template('list_files.html', files=files)
    except Exception as e:
        print(f'Ошибка при загрузке списка файлов: {e}')  # Логирование ошибки
        flash(f'Ошибка при загрузке списка файлов: {e}')
        return redirect(url_for('upload'))

@app.route('/analyze/<filename>')
@login_required
def analyze(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Начинаем анализ файла: {file_path}")  # Логирование

    try:
        # Открываем и читаем PDF файл
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        if not text:
            flash('Не удалось извлечь текст из PDF-файла.')
            return redirect(url_for('list_files'))

        # Используем SpaCy и TextRank для создания резюме текста
        doc = nlp(text)
        stopwords = list(spacy.lang.ru.stop_words.STOP_WORDS)
        word_frequencies = {}

        for word in doc:
            if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
                if word.text.lower() not in word_frequencies:
                    word_frequencies[word.text.lower()] = 1
                else:
                    word_frequencies[word.text.lower()] += 1

        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency

        sentence_scores = {}
        for sent in doc.sents:
            for word in sent:
                if word.text.lower() in word_frequencies:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

        summary_sentences = nlargest(5, sentence_scores, key=sentence_scores.get)
        final_summary = ' '.join([sent.text for sent in summary_sentences])

        print(f"Сгенерированное резюме: {final_summary}")  # Логирование

        # Сохранение резюме в отдельный файл
        summary_filename = f"summary_{os.path.splitext(filename)[0]}.txt"
        summary_file_path = os.path.join(app.config['SUMMARY_FOLDER'], summary_filename)
        with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write(final_summary)

        flash(f"Резюме успешно сохранено в файле {summary_filename}.")
        return render_template('analyze.html', summary=final_summary, filename=filename)

    except Exception as e:
        print(f'Ошибка при анализе файла: {e}')  # Логирование
        flash(f'Ошибка при анализе файла: {e}')
        return redirect(url_for('list_files'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Создание директорий uploads и summaries в корневой папке проекта
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SUMMARY_FOLDER'], exist_ok=True)
    
    with app.app_context():
        db.create_all()  # Инициализация базы данных
    
    app.run(debug=True)
