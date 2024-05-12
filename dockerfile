# Menggunakan image Python 3.12.3
FROM python:3.12.3

# Menyalin file requirements.txt ke dalam image
COPY requirements.txt /app/requirements.txt

# Mengatur working directory ke /app di dalam image
WORKDIR /app

# Menginstal pustaka-pustaka yang diperlukan
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh konten aplikasi ke dalam image
COPY . /app

# Menjalankan aplikasi Python
CMD ["python", "app.py"]