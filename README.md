# 🏗️ Construction Detector (SAM 3 / HuggingFace Transformers)

Агент, который по **видеофайлу** или **ссылке на YouTube** распознаёт и выделяет на
строительном объекте:

- 👷 **рабочих** (workers)
- 🚜 **технику** (excavator, bulldozer, loader, truck, crane, …)
- 🔢 **номерные знаки** техники (license plates)

В основе — модель **[SAM 3 от Meta](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)**
через **[🤗 Transformers](https://huggingface.co/docs/transformers/model_doc/sam3)**
(`Sam3Model` / `Sam3Processor`, чекпойнт `facebook/sam3`).

SAM 3 выполняет **Promptable Concept Segmentation (PCS)**: по короткой текстовой
фразе (например `"construction worker"`) он находит маски, рамки и оценки для
**каждого** объекта, подходящего под концепт. Видео обрабатывается покадрово.

---

## Как это устроено (по документации Transformers)

```python
from transformers import Sam3Model, Sam3Processor

model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
processor = Sam3Processor.from_pretrained("facebook/sam3")

inputs = processor(images=image, text="construction worker", return_tensors="pt").to(model.device)
outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs, threshold=0.5, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist(),
)[0]
# results -> {"masks": ..., "boxes": (xyxy, пиксели), "scores": ...}
```

Конвейер:

```
YouTube URL ──► yt-dlp ──┐
                         ├─► кадры ──► SAM 3 (текстовые промпты) ──► маски+боксы ──► видео
локальный файл ──────────┘
```

Чтобы не гонять тяжёлый vision-бэкбон на каждый промпт, на каждом кадре
vision-эмбеддинги считаются один раз (`model.get_vision_features(...)`) и
переиспользуются для всех концептов — как в разделе *«Efficient Multi-Prompt
Inference on Single Image»* документации.

| Файл | Назначение |
|------|------------|
| `main.py` | CLI-агент (точка входа) |
| `src/config.py` | Текстовые промпты концептов и цвета |
| `src/downloader.py` | Скачивание видео с YouTube (`yt-dlp`) |
| `src/sam3_segmenter.py` | Обёртка над `Sam3Model` / `Sam3Processor` |
| `src/annotator.py` | Отрисовка масок, рамок и подписей |
| `src/pipeline.py` | Связывает всё вместе, пишет итоговое видео |

---

## Требования

- **Transformers ≥ 5.12.0** (в этой версии появилась поддержка SAM 3).
- **PyTorch**, желательно с **CUDA** и **NVIDIA GPU** — модель работает на 1008px и
  тяжёлая. На CPU/Apple MPS запустится, но очень медленно (только для проверки на
  нескольких кадрах через `--max-frames`).
- Доступ к gated-репозиторию весов **`facebook/sam3`** на HuggingFace.

> 💡 Нет своей NVIDIA-видеокарты (например, Mac)? Запускайте на облачном GPU
> (Google Colab, RunPod, vast.ai и т. п.).

---

## Установка

### Windows

> Нужен **Python 3.10+** (рекомендуется 3.12). Скачать: https://www.python.org/downloads/
> (при установке отметьте галочку **«Add Python to PATH»**). Проверка: `py --version`.

```bat
cd construction_detector
setup.bat
venv\Scripts\activate
```

Затем получите доступ к весам:

```bat
REM 1) запросите доступ на https://huggingface.co/facebook/sam3
REM 2) залогиньтесь:
hf auth login
```

> 💡 Для склейки лучшего видео+аудио с YouTube пригодится **ffmpeg**:
> `winget install Gyan.FFmpeg` (или скачать с https://ffmpeg.org). Без него видео
> всё равно скачается, просто одним потоком.

### macOS / Linux

```bash
cd construction_detector
bash setup.sh
source venv/bin/activate
```

Затем получите доступ к весам:

```bash
# 1) запросите доступ на https://huggingface.co/facebook/sam3
# 2) залогиньтесь:
hf auth login        # или: huggingface-cli login
```

---

## Запуск

```bash
# Видео с YouTube, все три концепта
python main.py "https://www.youtube.com/watch?v=XXXX" -o out.mp4

# Локальный файл, только рабочие и техника
python main.py site.mp4 -o out.mp4 --concepts workers machinery

# Быстрая проверка: первые 30 кадров, каждый 3-й, на CPU
python main.py site.mp4 --frame-stride 3 --max-frames 30 --device cpu

# Встроенный демо-ролик
python main.py --demo -o demo_out.mp4
```

### Опции

| Опция | Описание |
|-------|----------|
| `source` | путь к файлу **или** ссылка YouTube |
| `--demo` | использовать встроенный демо-URL |
| `-o, --output` | путь к выходному видео (по умолч. `output.mp4`) |
| `--concepts` | `workers`, `machinery`, `license_plate` (можно несколько) |
| `--device` | `auto` / `cuda` / `mps` / `cpu` |
| `--score-thresh` | порог уверенности объекта (по умолч. `0.5`) |
| `--mask-thresh` | порог бинаризации маски (по умолч. `0.5`) |
| `--max-frames` | обработать не более N кадров |
| `--frame-stride` | обрабатывать каждый N-й кадр |

---

## Настройка под свою задачу

Промпты и цвета концептов задаются в [`src/config.py`](src/config.py). SAM 3 —
open-vocabulary, поэтому можно добавлять любые фразы: `"tower crane"`,
`"concrete mixer truck"`, `"safety helmet"`, `"reflective vest"` и т. д. Чем
конкретнее фраза, тем точнее сегментация.

---

## Примечание про видео

Документация Transformers описывает SAM 3 как **image PCS** (сегментация на
изображениях), поэтому видео мы прогоняем покадрово. У SAM 3 есть и видеотрекер
с устойчивыми ID объектов (см. репозиторий
[facebookresearch/sam3](https://github.com/facebookresearch/sam3)) — при
необходимости его можно подключить отдельно поверх этого же конвейера.
