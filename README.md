
Document Analyzer
Document Analyzer — это приложение для анализа, сравнения и управления документами. Оно позволяет извлекать текст из файлов PDF, генерировать краткие резюме с помощью OpenAI API, находить ключевую информацию (например, даты), сравнивать документы и управлять ими через удобный веб-интерфейс.

Основные функции
Генерация резюме:
Используется OpenAI GPT для автоматического создания краткого резюме текста документа.
Сравнение документов:
Подсветка общих слов и вычисление процента схожести с помощью метода TF-IDF.
Управление файлами:
Загрузка, просмотр и удаление файлов.
Синхронизация базы данных с загруженными файлами.
Поиск ключевых данных:
Автоматический поиск дат и другой информации в тексте документов.
Реестр документов:
Хранение информации о каждом документе, включая название, дату загрузки, резюме и найденные даты.
Управление пользователями:
Регистрация, авторизация и управление пользователями.
Установка
Клонируйте репозиторий:

bash
Копировать код
git clone https://github.com/ваш_пользователь/DocumentAnalyzer.git
cd DocumentAnalyzer
Установите зависимости:

bash
Копировать код
pip install -r requirements.txt
Настройте переменные окружения:

Создайте файл .env и добавьте:
makefile
Копировать код
OPENAI_API_KEY=ваш_ключ
Инициализируйте базу данных:

bash
Копировать код
flask db init
flask db migrate
flask db upgrade
Запустите сервер:

bash
Копировать код
flask run
Использование
Перейдите в браузере на http://127.0.0.1:5000.
Зарегистрируйтесь и войдите в приложение.
Загрузите PDF-документ через вкладку "Загрузить документ".
Перейдите в "Реестр документов", чтобы просмотреть сгенерированное резюме и найденные даты.
Используйте функцию "Сравнение документов", чтобы найти схожие фразы между двумя документами.
Удалите или синхронизируйте файлы через интерфейс.
Структура проекта
app.py — основной файл приложения Flask.
templates/ — HTML-шаблоны для веб-интерфейса.
static/ — статические файлы (CSS, JS).
uploads/ — директория для загруженных файлов.
summaries/ — директория для сгенерированных резюме.
models.py — модели базы данных.
Пример функционала
Реестр документов:
Название документа: example.pdf
Резюме: "Документ описывает ключевые аспекты проекта..."
Найденные даты: 01-01-2023
Сравнение документов:
Сходство между двумя файлами: 85%
Технологии
