# Menggunakan image Python 3.10.12-slim
FROM python:3.10.12-slim

# Install dependensi yang dibutuhkan oleh OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

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
