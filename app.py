from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask import send_from_directory
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import openai
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib  # Для сравнения текстов
import re  # Для поиска дат
from datetime import datetime
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# Пути к директориям uploads и summaries
base_dir = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads/')
app.config['SUMMARY_FOLDER'] = os.path.join(base_dir, 'summaries/')

db = SQLAlchemy(app)  # Инициализация базы данных
migrate = Migrate(app, db)

# Инициализация LoginManager
login_manager = LoginManager()
login_manager.init_app(app)  # Привязываем его к приложению Flask
login_manager.login_view = 'login'  # Указываем маршрут для страницы логина

openai.api_key = os.getenv('OPENAI_API_KEY', 'your_secrer_key')

# Таблица для хранения данных о документах
class Document(db.Model):
    __tablename__ = 'document'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    content = db.Column(db.Text, nullable=False)  # Текст документа
    summary = db.Column(db.Text, nullable=True)  # Сгенерированное резюме
    filename = db.Column(db.String(150), nullable=False)  # Имя файла
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    similarity = db.Column(db.Float, nullable=True)  # Процент схожести
    dates = db.Column(db.Text, nullable=True)  # Найденные даты



# Модель для пользователей
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Инициализация базы данных при запуске приложения
with app.app_context():
    db.drop_all() 
    db.create_all()  # Создание всех таблиц при запуске

# Маршруты приложения (анализ документов, сравнение, загрузка файлов и т.д.)
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('upload'))  # Перенаправляем авторизованного пользователя на загрузку файлов
    return redirect(url_for('login'))  # Неавторизованных на страницу логина

@app.route('/generate_summary/<int:document_id>')
@login_required
def generate_summary(document_id):
    document = Document.query.get_or_404(document_id)
    
    if not document:
        flash('Документ не найден.')
        return redirect(url_for('document_registry'))

    # Генерация резюме для документа
    try:
        text = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], document.filename))
        document.summary = generate_summary_with_openai(text)
        db.session.commit()
        flash(f'Резюме успешно создано для документа {document.filename}.')
    except Exception as e:
        flash(f'Ошибка при создании резюме: {e}')
    
    return redirect(url_for('document_registry'))

@app.route('/view_document/<int:document_id>')
@login_required
def view_document(document_id):
    document = Document.query.get_or_404(document_id)
    return render_template('view_document.html', document=document)

@app.route('/compare_selected_documents', methods=['POST'])
@login_required
def compare_selected_documents():
    # Получаем имена выбранных файлов из формы
    file1 = request.form['document1']
    file2 = request.form['document2']

    # Извлечение текста из выбранных файлов
    text1 = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], file1))
    text2 = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], file2))

    # Вычисление процента схожести
    similarity_percentage = calculate_similarity(text1, text2)

    # Подсветка общих слов
    highlighted_text1, highlighted_text2 = highlight_common_words(text1, text2)

    # Отправляем результат на страницу сравнения
    return render_template(
        'compare_files.html',
        file1=file1,
        file2=file2,
        text1=highlighted_text1,
        text2=highlighted_text2,
        similarity=similarity_percentage
    )


@app.route('/delete_document/<int:document_id>', methods=['POST'])
@login_required
def delete_document(document_id):
    document = Document.query.get_or_404(document_id)
    try:
        # Удаление файла из файловой системы
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], document.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Удаление записи из базы данных
        db.session.delete(document)
        db.session.commit()
        flash(f'Документ {document.filename} успешно удален.')
    except Exception as e:
        flash(f'Ошибка при удалении документа: {e}')
    
    return redirect(url_for('document_registry'))


# Функция для выделения общих слов
def highlight_common_words(text1, text2):
    matcher = difflib.SequenceMatcher(None, text1.split(), text2.split())
    result1 = []
    result2 = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal': 
            result1.append(f"<span class='highlight'>{' '.join(text1.split()[i1:i2])}</span>")
            result2.append(f"<span class='highlight'>{' '.join(text2.split()[j1:j2])}</span>")
        else:
            result1.append(' '.join(text1.split()[i1:i2]))
            result2.append(' '.join(text2.split()[j1:j2]))
    
    return ' '.join(result1), ' '.join(result2)

# Функция для вычисления процента схожести
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_percentage = similarity_matrix[0][0] * 100
    return similarity_percentage

# Функция для поиска дат в тексте
def find_dates_in_text(text):
    date_pattern = r'\b(?:\d{1,2}[./-])?(?:\d{1,2}[./-])?(?:\d{2,4})\b'
    dates = re.findall(date_pattern, text)
    return dates

# Функция для фильтрации и форматирования дат
def filter_and_format_dates(dates):
    valid_dates = []
    for date_str in dates:
        for fmt in ('%d-%m-%Y', '%d.%m.%Y', '%Y', '%m-%Y', '%Y-%m-%d'):
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                valid_dates.append(parsed_date)
                break
            except ValueError:
                continue
    return valid_dates

# Маршрут для отображения реестра документов
@app.route('/documents')
@login_required
def document_registry():
    documents = Document.query.all()  # Получаем все документы из базы данных
    return render_template('documents.html', documents=documents)

# Маршрут для анализа файлов
@app.route('/analyze/<filename>')
@login_required
def analyze(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"Начинаем анализ файла: {file_path}")

    try:
        text = extract_text_from_pdf(file_path)

        if not text:
            flash('Не удалось извлечь текст из PDF-файла.')
            return redirect(url_for('list_files'))

        # Генерация резюме с помощью OpenAI
        final_summary = generate_summary_with_openai(text)
        print(f"Сгенерированное резюме: {final_summary}")

        # Ищем даты в тексте
        raw_dates_found = find_dates_in_text(text)
        print(f"Найденные даты: {raw_dates_found}")

        # Фильтруем и форматируем даты
        formatted_dates = filter_and_format_dates(raw_dates_found)
        print(f"Отформатированные даты: {formatted_dates}")

        # Сохранение результатов анализа в базу данных
        document = Document(
            title=filename,
            content=text,
            filename=filename,
            summary=final_summary,
            dates=', '.join([date.strftime('%d-%m-%Y') for date in formatted_dates])
        )
        db.session.add(document)
        db.session.commit()

        flash(f"Резюме и найденные даты успешно сохранены.")
        return render_template('analyze.html', summary=final_summary, dates=formatted_dates, filename=filename)

    except Exception as e:
        print(f'Ошибка при анализе файла: {e}')
        flash(f'Ошибка при анализе файла: {e}')
        return redirect(url_for('list_files'))

# Функция для извлечения текста из PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        flash(f'Ошибка при извлечении текста из PDF файла: {e}')
        return ""

# Функция для генерации резюме с помощью OpenAI
def generate_summary_with_openai(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Сделай резюме для следующего текста: {text}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        print(f"Ошибка при создании резюме с помощью OpenAI: {e}")
        return "Ошибка при создании резюме."

# Маршрут для сравнения файлов с подсветкой совпадающих слов
@app.route('/compare_files', methods=['GET', 'POST'])
@login_required
def compare_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        file1 = request.form['file1']
        file2 = request.form['file2']

        # Извлечение текста из выбранных файлов
        text1 = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], file1))
        text2 = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], file2))

        # Вычисление процента схожести
        similarity_percentage = calculate_similarity(text1, text2)

        # Подсветка общих слов
        highlighted_text1, highlighted_text2 = highlight_common_words(text1, text2)

        return render_template(
            'compare_files.html', 
            files=files, 
            file1=file1, 
            file2=file2, 
            text1=highlighted_text1, 
            text2=highlighted_text2, 
            similarity=similarity_percentage
        )

    return render_template('compare_files.html', files=files)

# Регистрация пользователей
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

# Логин пользователей
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

# Загрузка файлов
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

            # Сохранение информации о документе в базе данных без лишних полей
            new_document = Document(
                title=file.filename,
                filename=file.filename,
                content=extract_text_from_pdf(file_path)
            )
            db.session.add(new_document)
            db.session.commit()

            flash('Файл успешно загружен.')
            return redirect(url_for('document_registry'))

    return render_template('upload.html')

@app.route('/sync_files')
@login_required
def sync_files():
    files_in_folder = os.listdir(app.config['UPLOAD_FOLDER'])  # Все файлы в папке
    existing_files = [doc.filename for doc in Document.query.all()]  # Файлы в базе данных

    new_files = [f for f in files_in_folder if f not in existing_files]  # Файлы, которых нет в базе

    for file in new_files:
        new_document = Document(
            title=file,
            filename=file,
            content='',  # Пустое поле для контента, если текст еще не извлечен
            summary='',  # Пустое поле для резюме, если оно еще не сгенерировано
            dates=''  # Пустое поле для дат
        )
        db.session.add(new_document)
    
    db.session.commit()
    flash('База данных синхронизирована с файлами в папке uploads.')
    return redirect(url_for('document_registry'))


# Список загруженных файлов
@app.route('/files')
@login_required
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])  # Получаем список всех файлов в папке
    return render_template('list_files.html', files=files)


# Выход пользователя
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Фавикон
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SUMMARY_FOLDER'], exist_ok=True)

    with app.app_context():
        db.create_all()

    app.run(debug=True)
