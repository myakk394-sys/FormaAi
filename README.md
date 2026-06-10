# FormaAi Studio: Unified 3D Neural Generation Pipeline

[English version](#english-version) | [Русская версия](#русская-версия)

---

## English Version

FormaAi is a unified hybrid 3D neural generation framework. It consolidates fast feed-forward generation, discrete 3D Gaussian Splatting representation, and continuous NeRF fields into a single, cohesive neural pipeline. Built on top of Microsoft's TRELLIS framework, it allows users to convert a single 2D image into high-fidelity textured 3D assets (GLB, PLY point clouds, and a compatible OBJ zip package) in seconds.

### Technical Architecture
The pipeline consists of the following processing stages:
1. **Preprocessing (Rembg)**: Background removal and image centering/resizing to a clean $518 \times 518$ RGB template.
2. **Stage 1 (Sparse Structure Flow Matching)**: Generates a sparse 3D occupancy lattice representing the coarse structure.
3. **Stage 2 (Structured Latent Flow Matching)**: Synthesizes structured latents corresponding to the object features.
4. **Model Offloading Decoders**: Decodes the structured latent into representation-specific formats (Mesh, Gaussians, Radiance Fields) sequentially on the GPU, offloading each to CPU memory immediately after to maintain a very low VRAM footprint.
5. **AI Texture Upscaling (Swin2SR)**: Uses Swin2SR models to upscale baked textures from $1024 \times 1024$ to 2K or 4K resolution dynamically. Includes an intelligent GPU OOM auto-fallback to CPU.

---

### Local Installation Guide

#### 1. System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (Minimum: **8 GB VRAM** for RTX 3060/4060; Recommended: **12+ GB VRAM**).
- **RAM**: 16 GB minimum (32 GB recommended for x4 texture upscaling).

#### 2. Virtual Environment Setup
Ensure Python 3.10 and virtualenv are installed. Create and activate the environment:
```bash
# Inside the project root folder
python3.10 -m venv TRELLIS-main/venv_trellis
source TRELLIS-main/venv_trellis/bin/source/activate
```

#### 3. Installing Dependencies
Install core dependencies, PyTorch, and specific CUDA sub-modules:
```bash
# 1. Install standard requirements
./TRELLIS-main/venv_trellis/bin/pip install -r requirements.txt

# 2. Install specialized CUDA packages (e.g. spconv, diff-gaussian-rasterization, xformers)
# Ensure CUDA_HOME is set if compiling from source:
export CUDA_HOME=/usr/local/cuda
./TRELLIS-main/venv_trellis/bin/pip install spconv-cu121
./TRELLIS-main/venv_trellis/bin/pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Install diff-gaussian-rasterization
./TRELLIS-main/venv_trellis/bin/pip install "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
```

#### 4. How to Start the Web UI
Run the FastAPI backend server using the virtual environment:
```bash
./TRELLIS-main/venv_trellis/bin/python -m uvicorn server:app --host 127.0.0.1 --port 7860
```
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) in your web browser.

#### 5. How to Run Standalone Tests
Execute the console verification script:
```bash
./TRELLIS-main/venv_trellis/bin/python test_formaai.py
```

---

## Русская Версия

**FormaAi** — это унифицированная гибридная нейросетевая среда генерации 3D-объектов. Она объединяет быструю генерацию методом прямого прохода (feed-forward), дискретное представление 3D Gaussian Splatting и непрерывные поля излучения (NeRF) в единый пайплайн. Созданный на базе SOTA-фреймворка TRELLIS от Microsoft, FormaAi преобразует одно 2D-изображение в высокодетализированный 3D-ассет (GLB-меш, PLY-облако точек и OBJ-пакет в архиве) за несколько секунд.

### Техническая Архитектура
Пайплайн состоит из следующих вычислительных этапов:
1. **Препроцессинг (Rembg)**: Автоматическое вырезание фона, центрирование и масштабирование картинки под шаблон $518 \times 518$ RGB.
2. **Этап 1 (Sparse Structure Flow Matching)**: Генерация разреженной пространственной решетки заполнения, описывающей грубую форму объекта.
3. **Этап 2 (Structured Latent Flow Matching)**: Синтез структурированного латентного представления для детальных признаков.
4. **Модульная выгрузка декодеров (Model Offloading)**: Декодирование латентов в Меш, Облако Гауссианов и Поле излучения происходит на GPU последовательно. Сразу после работы каждый декодер выгружается в ОЗУ (на CPU), что снижает пиковое потребление видеопамяти.
5. **AI-масштабирование текстур (Swin2SR)**: Использование Swin2SR для апскейлинга текстуры с базовых $1024 \times 1024$ до 2K/4K разрешения. Имеет систему автоматического перенаправления вычислений на CPU при нехватке VRAM (OOM).

---

### Инструкция по локальному запуску

#### 1. Системные требования
- **ОС**: Linux (рекомендуется Ubuntu 20.04+)
- **GPU**: Видеокарта NVIDIA с поддержкой CUDA (Минимум: **8 ГБ VRAM**; Рекомендуется: **12+ ГБ VRAM**).
- **ОЗУ**: Минимум 16 ГБ RAM (рекомендуется 32 ГБ для текстур 4K).

#### 2. Настройка виртуального окружения
Убедитесь, что установлены Python 3.10 и пакет venv. Создайте и активируйте окружение:
```bash
# В корневой папке проекта
python3.10 -m venv TRELLIS-main/venv_trellis
source TRELLIS-main/venv_trellis/bin/activate
```

#### 3. Установка зависимостей
Установите основные зависимости, PyTorch и специализированные CUDA-библиотеки:
```bash
# 1. Установка стандартных требований
./TRELLIS-main/venv_trellis/bin/pip install -r requirements.txt

# 2. Установка специализированных CUDA-пакетов
export CUDA_HOME=/usr/local/cuda
./TRELLIS-main/venv_trellis/bin/pip install spconv-cu121
./TRELLIS-main/venv_trellis/bin/pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Установка растеризатора Гауссианов
./TRELLIS-main/venv_trellis/bin/pip install "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
```

#### 4. Как запустить веб-интерфейс
Запустите FastAPI backend-сервер из виртуального окружения:
```bash
./TRELLIS-main/venv_trellis/bin/python -m uvicorn server:app --host 127.0.0.1 --port 7860
```
Откройте [http://127.0.0.1:7860](http://127.0.0.1:7860) в вашем браузере.

#### 5. Как запустить консольный тест
Запустите скрипт проверки в терминале:
```bash
./TRELLIS-main/venv_trellis/bin/python test_formaai.py
```
