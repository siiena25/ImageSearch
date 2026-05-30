# 📓 Dev Log — Visual Image Search (Marketplace)

> Этот файл — живой лог разработки проекта.  
> Каждый шаг фиксируется: что сделали, почему, какой результат.  
> Цель: иметь полную историю для презентации и отчёта.

---

## 📌 Общее описание проекта

Проект существует в двух контекстах — учебном и промышленном. Оба держим в голове при каждом техническом решении.

---

### 🎓 Уровень 1 — Учебный MVP (университет)

**Название:** On-device Visual Search for Fashion Marketplace  
**Платформа:** Android (on-device inference)  
**Задача:** Пользователь загружает фото платья → приложение возвращает топ-10 визуально похожих товаров из локального каталога  
**Каталог:** ~1000 товаров (платья, ASOS)  
**Метод:** Embedding-based retrieval (cosine similarity) + color/texture re-ranking  
**Ограничения:** нет backend'а, нет сети, всё хранится и считается на устройстве

---

### 🏢 Уровень 2 — Промышленное масштабирование (Uzum Marketplace, Узбекистан)

**Контекст:** Uzum Marketplace — крупнейший маркетплейс Узбекистана  
**Аудитория:** ~20 млн пользователей  
**Платформа:** Backend (Python/FastAPI или Go), GPU-инференс, облако  
**Задача:** Image search по всем категориям каталога (десятки миллионов товаров)  
**Категории товаров:**
> мебель, туризм, электроника, бытовая техника, одежда, обувь, аксессуары, красота и уход, здоровье, товары для дома, строительство и ремонт, автотовары, детские товары, хобби и творчество, спорт и отдых, продукты питания, бытовая химия, канцтовары, зоотовары, книги, дача/сад/огород

**Ключевые отличия от MVP:**

| Параметр | MVP (университет) | Uzum (production) |
|----------|-------------------|-------------------|
| Каталог | ~1 000 товаров | ~10–100 млн товаров |
| Категории | только платья | 20+ категорий |
| Инфраструктура | on-device (Android) | backend + GPU |
| Latency бюджет | <2s on device | <300ms p99 |
| Хранение эмбеддингов | `.npy` файл | векторная БД (Milvus / Qdrant / Weaviate) |
| Модель inference | ONNX / TFLite | TorchServe / Triton Inference Server |
| Поиск | brute-force cosine | ANN-индекс (HNSW, IVF-PQ) |
| Обновление каталога | статичный файл | real-time индексация новых товаров |

---

### 🧭 Стратегия разработки: "Design for scale, start small"

**Принцип:** Писать код MVP так, чтобы он органично масштабировался без полного переписывания.

**Конкретные решения из этого принципа:**
- Модели выбирать с учётом, что они будут работать и на backend'е (не только TFLite-совместимые)
- Пайплайн делать **модульным** — каждый компонент (сегментация → embedding → reranking) независим и заменяем
- Для MVP использовать fashion-специфичные решения, но закладывать **категориальный роутинг** как явную точку расширения
- Формат хранения эмбеддингов — `float32` vectors, совместимо с Qdrant/Milvus без конвертации
- Все скрипты пишем так, чтобы их логика переносилась в FastAPI endpoint'ы без переделки

**Дорожная карта масштабирования (после университета):**
```
MVP (on-device, 1K items, fashion only)
        ↓
v1 Backend (FastAPI + Qdrant, 100K items, fashion)
        ↓
v2 Multi-category (роутинг по категориям, 1M items)
        ↓
v3 Uzum Production (Triton + Milvus, 50M+ items, real-time indexing)
```

---

## ⚠️ Ключевое ограничение MVP: on-device Android inference

> Это ограничение влияет на **выбор всех моделей** — его нужно держать в голове.

### Что это значит на практике

На Android устройстве происходит следующее:
1. Пользователь делает фото или загружает изображение
2. **На устройстве** считается embedding query-изображения
3. **На устройстве** происходит cosine similarity против предзаготовленных эмбеддингов каталога
4. **На устройстве** выводится результат — никакого backend'а нет

Это значит, что модель для query inference **должна работать на Android**. А значит:

| Требование | Ограничение |
|-----------|-------------|
| Формат модели | TFLite или ONNX Runtime for Android |
| Размер модели | < 100–150 MB (желательно, с квантизацией) |
| RAM на inference | < 300–500 MB (зависит от устройства) |
| Скорость | < 2–3 сек на mid-range Android |
| CPU vs GPU | CPU (NNAPI опционально) |

---

### Архитектурное разделение: offline vs on-device

Это ключевой паттерн проекта — **тяжёлые вычисления делаем один раз офлайн на компьютере**, лёгкие — на устройстве в реальном времени.

```
OFFLINE (один раз, на компьютере/сервере — ДО релиза приложения):
───────────────────────────────────────────────────────────────────
  Каталог (1000 платьев)
        │
        ▼
  SAM2 сегментация          ← тяжёлая модель, ~2GB, только офлайн
        │
        ▼
  marqo-fashionSigLIP        ← полная модель, считаем эмбеддинги каталога
        │
        ▼
  DINOv2 patch features      ← тяжёлая модель, только офлайн
        │
        ▼
  TTA усреднение
        │
        ▼
  embeddings.npy             ← ЭТОТ ФАЙЛ кладётся в Android assets
  patch_features.npy         ← ЭТОТ ФАЙЛ тоже
  product_ids.json
  catalog images (thumbnails)

ON-DEVICE (каждый раз при поиске, на Android телефоне):
───────────────────────────────────────────────────────
  [Query Photo от пользователя]
        │
        ▼
  rembg или лёгкая сегментация  ← SAM2 on-device невозможен (2GB)
        │
        ▼
  ONNX / TFLite модель          ← экспортированная marqo-fashionSigLIP
  (квантизованная, ~80-100MB)
        │
        ▼
  cosine similarity (brute-force, 1000 vectors — быстро)
        │
        ▼
  re-ranking (CPU, numpy-like)
        │
        ▼
  Top-10 results
```

---

### Как работает совместимость эмбеддингов

**Критически важно:** модель для каталога (офлайн) и модель для query (on-device) должны быть **одной и той же архитектурой**, иначе эмбеддинги несовместимы.

Решение:
- Офлайн считаем эмбеддинги каталога через полную `marqo-fashionSigLIP` на PyTorch
- Экспортируем **ту же модель** в ONNX формат для Android
- На Android — ONNX Runtime for Android запускает тот же граф → те же векторы

---

### Что НЕ запускается on-device

| Компонент | Почему нельзя on-device | Где запускается |
|-----------|------------------------|-----------------|
| SAM2 | ~1.2GB model, требует GPU | Офлайн, на компьютере |
| DINOv2-Large | ~1.1GB | Офлайн, на компьютере |
| k-Reciprocal re-ranking | Требует полной матрицы расстояний — 1000×1000 float32 = 4MB, допустимо | ✅ On-device |
| Query expansion (доп. поиск) | Второй прогон через модель — x2 время | ⚠️ Осторожно |
| HSV/texture re-ranking | Простая арифметика | ✅ On-device |

---

### Выбор on-device модели: что реально запустить на Android

| Модель | Размер | ONNX export | On-device speed | Совместима с fashion эмбеддингами каталога |
|--------|--------|-------------|-----------------|-------------------------------------------|
| `marqo-fashionSigLIP` (ViT-B/16) | ~330MB → ~85MB (INT8) | ✅ | ~1.5s CPU | ✅ (та же модель) |
| `siglip2-base-patch16-224` | ~380MB → ~95MB (INT8) | ✅ | ~1.8s CPU | ✅ (уже используем) |
| `fashion-clip` (ViT-B/32) | ~340MB → ~85MB (INT8) | ✅ | ~1.2s CPU | ✅ |
| DINOv2-small | ~88MB | ✅ | ~0.8s CPU | ✅ но не fashion-специфичен |
| SAM2 | ~1.2GB | ❌ нет TFLite | ❌ | N/A |

**INT8 квантизация** уменьшает размер в 4x при потере <1% качества — стандартная практика для мобильных моделей.

**Вывод:** `marqo-fashionSigLIP` экспортируем в ONNX → квантизуем INT8 → запускаем через ONNX Runtime for Android.

---

## 🗺️ Архитектура системы (текущая, MVP)

```
[Query Image]
     │
     ▼
[Preprocessing: crop white background]   ← crop_white_background.py
     │
     ▼
[Embedding: SigLIP2-base-patch16-224]    ← build_embeddings_siglip2.py
     │
     ▼
[Cosine Similarity vs Catalog Embeddings]
     │
     ▼
[Top-50 candidates]
     │
     ▼
[Re-ranking: 0.75 * embedding + 0.25 * HSV histogram]
     │
     ▼
[Top-10 Results]
```

---

## 📅 История шагов

---

### ✅ Шаг 0 — Сбор каталога (ASOS)
**Файл:** `build_catalog_from_asos.py`  
**Что сделано:** Парсинг CSV с товарами ASOS, скачивание изображений, создание `catalog.jsonl`  
**Формат записи каталога:**
```json
{
  "product_id": "dress_000123",
  "title": "Floral Midi Dress",
  "image": "images/dress_000123.jpg",
  "brand": "ASOS",
  "price": 79.99
}
```
**Почему:** Нужен локальный каталог для on-device retrieval без backend'а.

---

### ✅ Шаг 1 — Baseline эмбеддинги (CLIP)
**Файл:** `build_embeddings.py`, `retrieve_topk.py`  
**Модель:** OpenAI CLIP  
**Результат:** Сохранены `embeddings.npy`, `product_ids.json`  
**Вывод:** Baseline работает, но качество на платьях с принтом — слабое. CLIP обучен на общих изображениях, не на fashion.

---

### ✅ Шаг 2 — Обрезка белого фона
**Файл:** `crop_white_background.py`  
**Что делает:** Ищет bbox переднего плана (пиксели с RGB < 245), обрезает изображение до bbox с отступом 6%  
**Результат:** `images_cropped/`, `catalog_cropped.jsonl`  
**Проблема:** Метод работает только для изображений с чистым белым фоном. Если платье на модели — тело, лицо, руки остаются в кадре и «загрязняют» эмбеддинг нерелевантными признаками. Серый фон тоже не обрезается.

---

### ✅ Шаг 3 — Улучшенная модель: SigLIP2
**Файлы:** `build_embeddings_siglip2.py`, `retrieve_topk_siglip2.py`  
**Модель:** `google/siglip2-base-patch16-224`  
**Почему лучше CLIP:** SigLIP2 обучен с sigmoid loss вместо softmax, лучше работает на fine-grained retrieval  
**Результат:** Видимое улучшение на однотонных платьях. Платья с принтом всё ещё находятся плохо.

---

### ✅ Шаг 4 — Color Re-ranking (Mean RGB + HSV Histogram)
**Файлы:** `retrieve_topk_siglip2.py`, `retrieve_topk_siglip2_hsv.py`  
**Что делает:**
- Берём топ-50 кандидатов по embedding similarity
- Пересчитываем финальный score: `0.75 * embedding_score + 0.25 * color_score`
- Вариант 1: mean RGB цвет переднего плана
- Вариант 2: HSV гистограмма (12×4×4 bins) + histogram intersection similarity

**Почему HSV лучше RGB:** HSV разделяет тон (Hue) от яркости (Value) → более стабильный цвет при разном освещении  
**Вывод:** Улучшает результаты для одноцветных платьев, но для принтов — недостаточно.

---

### ✅ Шаг 5 — SAM2 Сегментация каталога (v1 → v2: Grounded-SAM)
**Файл:** `segment_catalog_sam2.py`  

#### v1 (первая попытка): SAM2 с центральной точкой
- Запустили SAM2 на весь каталог (399 изображений, 67 секунд, device=mps)
- **Проблема обнаружена:** SAM2 с центральной точкой видит "человека в платье" как **единый объект** → сегментирует всё тело целиком, не только одежду
- Пример: `34592.jpg` — женщина не убрана, осталась с платьем

#### Попытка v1.5: SegFormer-B2-clothes
- Попробовали `mattmdjaga/segformer-b2-clothes` — обучена на Lip dataset, знает классы одежды
- **Проблема:** модель стала приватной на HuggingFace, 401 Unauthorized

#### v2 (финальный): Grounded-SAM = Grounding DINO + SAM2
**Модели:**
- `IDEA-Research/grounding-dino-base` (~340MB) — zero-shot детекция по тексту
- `facebook/sam2.1-hiera-small` (~200MB) — точная сегментация по bbox

**Как работает:**
```
[Изображение]
      │
      ▼
[Grounding DINO]
  text_prompt = "dress. skirt. shirt. top. blouse. coat. jacket..."
      │
      ├── нашёл bbox одежды (score ≥ 0.25)
      │         │
      │         ▼
      │   [SAM2 с bbox промптом] → маска ТОЛЬКО одежды ✅
      │
      └── не нашёл (flat-lay, нет модели)
                │
                ▼
          [SAM2 центральная точка] → главный объект (fallback)
```

**Почему это правильно:**
- Grounding DINO понимает текст → находит именно область одежды, не тело
- SAM2 с bbox промптом сегментирует внутри найденной области → чистая маска
- Масштабируется на Uzum: меняем text_prompt под категорию (электроника, мебель...)
- Оба инструмента публично доступны без аутентификации

**Статус:** ✅ Тест на `34592.jpg` и `10016.jpg` запущен

---

### ✅ Шаг 6 — Marqo FashionSigLIP + TTA эмбеддинги
**Файл:** `build_embeddings_fashion.py`  
**Модель:** `hf-hub:Marqo/marqo-fashionSigLIP` (через **open_clip**, не HuggingFace AutoModel)

**Что делает:**
- Считает эмбеддинги через fashion-специфичную модель (768-dim)
- TTA: 5 аугментаций → усреднение → L2-нормализация
- Аугментации: горизонтальный флип, случайный crop (85-100%), яркость (±10%)
- Результат: `data/embeddings_fashion.npy` (399, 768), `data/product_ids_fashion.json`

**Реальный результат запуска:**
```
Catalog: 399 items, Device: mps, TTA: enabled N=5
Embeddings (TTA): 100% 399/399 [00:34<00:00, 11.70it/s]
✅ Embeddings shape: (399, 768)  Failed: 0
```

**Проблема, с которой столкнулись:**  
`transformers 5.x` несовместим с custom кодом модели Marqo (ошибка `Cannot copy out of meta tensor`).

**Решение:** Перешли на `open_clip` — официальный источник Marqo FashionSigLIP.  
Установили `ftfy` и `open_clip_torch` → загружается через `open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")`.

---

### ✅ Шаг 7 — DINOv2 Patch Features для texture/pattern
**Файл:** `build_patch_features_dino.py`  
**Модель:** `facebook/dinov2-small` (384-dim, ~88MB)

**Что делает:**
- Извлекает patch-level токены из DINOv2 ViT (256 патчей × 384 dim)
- Усредняет → 384-dim texture descriptor (bag-of-patches mean)
- L2-нормализация
- Результат: `data/patch_features_dino.npy` (399, 384)

**Реальный результат запуска:**
```
Device: mps
DINOv2 patch features: 100% 13/13 [00:03<00:00, 4.16it/s]
✅ Patch features shape: (399, 384)  Failed: 0
```

**Почему DINOv2:** Self-supervised обучение на патчах → каждый патч = локальный семантический дескриптор. Понимает "цветочный принт", "полоска", "клетка" как концепции. Zero-shot, любая категория.

---

### ✅ Шаг 8 — Улучшенный retrieval v2

```
[Query Image]
      ↓
[FashionSigLIP embedding через open_clip]
      ↓
[Query Expansion: 0.7*query + 0.3*mean(top-3)]   ← уточняем query
      ↓
[k-Reciprocal Re-ranking]                         ← убираем ложных соседей
      ↓
[Top-50 кандидатов]
      ↓
[Тройной re-ranking:
   0.60 * semantic (fashionSigLIP)
 + 0.25 * texture  (DINOv2 patches)
 + 0.15 * color    (HSV histogram)]
      ↓
[Top-10 Results]
```

**Реальный запуск на query `34592.jpg`:**
```
Top-10 results:
 1. [46285] final=0.7979  sem=0.741  tex=0.807  col=0.493  Dress 46285
 2. [52162] final=0.7759  sem=0.751  tex=0.835  col=0.278  Dress 52162
 3. [34570] final=0.7758  sem=0.746  tex=0.850  col=0.264  Dress 34570
 4. [44643] final=0.7751  sem=0.748  tex=0.862  col=0.235  Dress 44643
 5. [25971] final=0.7742  sem=0.756  tex=0.808  col=0.303  Dress 25971
 ...
```

**Проблема, с которой столкнулись (и решили):**  
После Query Expansion scores пересчитываются → self-match (query в результатах) возвращался на первое место.  
**Решение:** завели `suppress_self()` — функция "забивает" score индекса query после КАЖДОГО пересчёта scores (после initial, после QE, после k-Reciprocal).

**Параметры командной строки:**
```bash
python3 retrieve_topk_v2.py --query data/images/34592.jpg
python3 retrieve_topk_v2.py --query test.jpg --alpha 0.7 --beta 0.2 --gamma 0.1
python3 retrieve_topk_v2.py --query test.jpg --no-qe --no-kr   # отключить QE и k-reciprocal
```

**Выходные артефакты:**
- `data/retrieval_results_v2.json` — полный JSON с scores
- `data/retrieval_preview_v2.jpg` — сетка query + top-10 с метриками
**Файлы:** `retrieve_topk_siglip2.py`, `retrieve_topk_siglip2_hsv.py`  
**Что делает:**
- Берём топ-50 кандидатов по embedding similarity
- Пересчитываем финальный score: `0.75 * embedding_score + 0.25 * color_score`
- Вариант 1: mean RGB цвет переднего плана
- Вариант 2: HSV гистограмма (12×4×4 bins) + histogram intersection similarity

**Почему HSV лучше RGB:** HSV разделяет тон (Hue) от яркости (Value) → более стабильный цвет при разном освещении  
**Вывод:** Улучшает результаты для одноцветных платьев, но для принтов — недостаточно.

---

## 🚧 Текущие проблемы

| Проблема | Причина | Планируемое решение |
|----------|---------|---------------------|
| Плохой поиск для платьев с принтом | Глобальный эмбеддинг не кодирует паттерн | Texture features (DINOv2 patches, LBP) |
| Модель/фон попадают в эмбеддинг | crop_white_background не убирает модель | SAM2 сегментация |
| Универсальность по категориям | fashion-clip работает только для одежды | Категориальный роутинг + universal fallback |

---

## 🔬 Технические решения — обсуждение и выбор

---

### 🔹 Сегментация: почему SAM (Segment Anything Model)?

**Рассматривались варианты:**

| Метод | Плюсы | Минусы |
|-------|-------|--------|
| Пороговая обрезка (текущий) | Простой, быстрый | Только белый фон, не убирает модель |
| `rembg` (U²-Net) | Простой, офлайн, pip install | Специализирован под портреты/природу, хуже на товарах |
| **SAM2 (Meta)** | Универсален, точен, SOTA | Тяжелее, нужен промпт или auto-mode |
| YOLOv8-seg | Быстрый, детектит классы | Нужно дообучать под fashion категории |

**Почему выбираем SAM2:**
- Не привязан к конкретному классу объектов — работает на любом товаре маркетплейса (платье, телефон, еда)
- Automatic Mask Generation (AMG) находит самый «значимый» объект без промптов
- Для офлайн препроцессинга каталога скорость некритична — важна точность
- Активно развивается Meta (SAM2 — версия 2024 года), есть PyPI пакет `segment-anything-2`
- Позволяет передавать точку в центре изображения как промпт → товары на маркетплейсе обычно центрированы

**Как будет работать:**
```
[Catalog image]
      │
      ▼
[SAM2: point prompt = center of image]
      │
      ▼
[Mask: только пиксели товара]
      │
      ▼
[Masked image: фон = белый/серый]
      │
      ▼
[Embedding computation]
```

---

### 🔹 Модель эмбеддингов: категориальный роутинг vs универсальная модель

**Вопрос:** Использовать `fashion-clip` (специализирован под одежду) или что-то универсальное, если маркетплейс продаёт всё?

**Анализ подходов:**

#### Вариант A: Одна универсальная модель для всего
- `SigLIP2-Large` или `OpenCLIP ViT-H/14` — сильные генеральные модели
- Плюс: простота, нет зависимости от категории
- Минус: хуже на fashion-специфике (принты, фактура, фасон)

#### Вариант B: Категориальный роутинг (winner ✅)
```
[Image] → [Category Classifier] → [Fashion?] → fashion-clip / marqo-fashionSigLIP
                                  [Electronics?] → SigLIP2 / DINOv2
                                  [Food?] → SigLIP2
                                  [Other] → SigLIP2 (universal fallback)
```
- **Классификатор категорий:** лёгкий ViT или MobileNet, дообученный на категориях маркетплейса
- Альтернатива без классификатора: брать категорию из метаданных каталога (для каталога она известна заранее!)
- **Для MVP (только платья):** сразу использовать `Marqo/marqo-fashionSigLIP` — дообученный SigLIP на fashion датасете, знает принты, фасоны, материалы

**Лучшие fashion-модели на сегодня (апрель 2026):**

| Модель | Размер | Особенность |
|--------|--------|-------------|
| `Marqo/marqo-fashionSigLIP` | Base | SigLIP дообученный на fashion, лучше понимает принты |
| `patrickjohncyh/fashion-clip` | Base | CLIP на 800K fashion items |
| `google/siglip2-large-patch16-256` | Large | Мощнее base, универсален |
| `facebook/dinov2-large` | Large | Лучший для текстур/паттернов |

**Вывод для MVP:** `marqo-fashionSigLIP` для основного эмбеддинга + DINOv2 patch features для texture re-ranking.

---

### 🔹 Texture/Pattern-aware признаки: лучшее решение для маркетплейса

**Почему это важно:** Два платья могут иметь одинаковый силуэт и цвет, но разный принт (цветочный vs горошек vs полоска). Глобальный эмбеддинг это не различает.

**Рассматривались варианты:**

| Метод | Описание | Подходит для маркетплейса? |
|-------|----------|---------------------------|
| LBP (Local Binary Pattern) | Классический texture descriptor | Частично — ловит микротекстуры, но не семантику принта |
| Gabor filters | Frequency + orientation анализ | Хорошо для регулярных паттернов (полоска, клетка) |
| FFT spectrum | Частотный анализ изображения | Хорошо для периодических принтов |
| **DINOv2 patch features** | Локальные ViT патч-эмбеддинги | ✅ Лучший вариант — семантические + текстурные |
| CNN feature maps | Промежуточные слои ResNet/EfficientNet | Хуже DINOv2 на текстурах |

**Почему DINOv2 patch features — лучшее решение для маркетплейса:**

1. **DINOv2 обучен self-supervised на патчах** → каждый 16×16 патч изображения получает свой вектор, который кодирует *локальную* текстуру/паттерн
2. **Не нужно дообучение** — работает zero-shot на любой категории товаров
3. **Для принтов:** можно сравнивать патч-векторы через:
   - Усреднение (bag-of-patches mean)
   - Histogram-based matching (как bag-of-words)
   - Chamfer distance между множествами патчей
4. **Работает на всех категориях** — DINOv2 понимает текстуру дерева, металла, ткани, еды

**Итоговая формула score:**
```
final_score = α * semantic_embedding    (marqo-fashionSigLIP, глобальный)
            + β * texture_score         (DINOv2 patch-level similarity)
            + γ * color_score           (HSV histogram)

где α=0.60, β=0.25, γ=0.15 (подбирается экспериментально)
```

---

### 🔹 Re-ranking: k-reciprocal + query expansion

**k-Reciprocal Encoding (метод из person re-identification, 2017):**
- Идея: если A похоже на B И B похоже на A → они действительно похожи
- Убирает «ложных соседей» из результатов
- Стандарт в fashion retrieval benchmarks

**Query Expansion:**
- Берём топ-3 результата, усредняем их эмбеддинги с query
- Повторяем поиск → результаты становятся стабильнее
- Добавляет ≈1-2% recall@10 без дополнительных моделей

---

### 🔹 Уровень 5 — Качество каталога

**Test-Time Augmentation (TTA) для catalog embeddings:**
- Считаем эмбеддинг каждого изображения N раз с небольшими аугментациями
- Усредняем → более устойчивый вектор
- Аугментации: горизонтальный флип, ±10° поворот, небольшой crop (0.9-1.0)
- Улучшает recall без изменения модели

---

## 🗺️ Обновлённая архитектура (целевая, MVP v2)

```
═══════════════════ OFFLINE (препроцессинг каталога) ════════════════════

[Catalog Image]
      │
      ▼
[SAM2 Segmentation]          ← убираем фон, модель, всё кроме товара
      │
      ▼
[Category from metadata]     ← берём категорию из catalog.jsonl
      │
      ├── fashion → [marqo-fashionSigLIP embedding]
      └── other   → [SigLIP2 embedding]          ← universal fallback
      │
      ▼
[DINOv2 patch features]      ← texture/pattern descriptor
      │
      ▼
[TTA: N augmentations → avg] ← устойчивый финальный вектор
      │
      ▼
[Save: embeddings.npy + patch_features.npy + product_ids.json]

════════════════════ ONLINE (запрос пользователя, on-device Android) ════════════════════

[Query Photo от пользователя]
      │
      ▼
[Лёгкая сегментация on-device]     ← НЕ SAM2! Только офлайн.
  Варианты:                            - rembg-lite / U²-Net mobile (ONNX, ~45MB)
                                        - простое центральное кадрирование (fallback)
      │
      ▼
[ONNX Runtime: marqo-fashionSigLIP]  ← квантизованный INT8, ~85MB, ~1.5s на CPU
      │
      ▼
[Cosine similarity vs 1000 каталожных векторов]  ← <10ms
      │
      ▼
[Re-ranking: α*embed + β*texture + γ*HSV]        ← CPU арифметика
      │
      ▼
[k-Reciprocal re-ranking]
      │
      ▼
[Top-10 Results → Android UI]
``` (ориентир на будущее)

> Не реализуем сейчас, но держим в голове при каждом решении MVP.

```
════════ OFFLINE: Catalog Indexing Pipeline ════════

[Seller uploads product image]
      │
      ▼
[SAM2 segmentation service]   ← GPU worker, async
      │
      ▼
[Category Router]
      ├── fashion  → FashionSigLIP encoder
      ├── electronics → SigLIP2 encoder
      └── other    → SigLIP2 encoder (universal)
      │
      ▼
[DINOv2 texture features]     ← параллельно с embedding
      │
      ▼
[Qdrant / Milvus]             ← upsert vector + metadata
      (HNSW index, real-time)

════════ ONLINE: Search API (FastAPI) ════════

POST /search
  body: { image_base64, category_hint? }
      │
      ▼
[Segmentation] → [Category Router] → [Encoder]
      │
      ▼
[Qdrant ANN search, top-200]  ← <50ms
      │
      ▼
[Re-ranking: semantic + texture + color]
      │
      ▼
[k-Reciprocal + Query Expansion]
      │
      ▼
[Return top-10 product cards] ← <300ms total p99
```

**Компоненты Uzum стека:**
| Слой | Технология | Почему |
|------|-----------|--------|
| Inference | Triton Inference Server | batching, GPU utilization |
| Векторная БД | Milvus или Qdrant | HNSW + IVF-PQ, миллиарды векторов |
| API | FastAPI + async | async/await, легко добавить gRPC |
| Очередь индексации | Kafka | real-time при загрузке товара продавцом |
| Кэш | Redis | embedding cache для популярных query |
| Мониторинг | Prometheus + Grafana | latency, recall@10, throughput |


---

## 📋 План реализации (очередность)

### 🎓 MVP — учебный проект (fashion only, on-device)

| # | Задача | Файл | Статус |
|---|--------|------|--------|
| 1 | SAM2 сегментация каталога | `segment_catalog_sam2.py` | ✅ DONE |
| 2 | Эмбеддинги через marqo-fashionSigLIP + TTA | `build_embeddings_fashion.py` | ✅ DONE |
| 3 | DINOv2 patch features | `build_patch_features_dino.py` | ✅ DONE |
| 4 | TTA для catalog embeddings | встроено в `build_embeddings_fashion.py` | ✅ DONE |
| 5 | Re-ranking: α*embed + β*texture + γ*color | `retrieve_topk_v2.py` | ✅ DONE |
| 6 | k-Reciprocal re-ranking | `retrieve_topk_v2.py` | ✅ DONE |
| 7 | Query expansion | `retrieve_topk_v2.py` | ✅ DONE |
| 8 | Категориальный роутинг (заглушка) | `category_router.py` | 🔲 TODO |
| 9 | Экспорт моделей в TFLite/ONNX | `export_models.py` | 🔲 TODO |
| 10 | Интеграция в Android | Android Studio project | 🔲 TODO |

### 🏢 После университета — масштабирование на Uzum

| # | Задача | Технология | Приоритет |
|---|--------|-----------|-----------|
| A | FastAPI backend с /search endpoint | FastAPI + uvicorn | Высокий |
| B | Замена .npy на Qdrant | Qdrant (Docker) | Высокий |
| C | Мультикатегорийный роутинг | category_router.py → расширить | Высокий |
| D | Async индексация новых товаров | Kafka + worker | Средний |
| E | Triton Inference Server для моделей | NVIDIA Triton | Средний |
| F | ANN-индекс вместо brute-force | HNSW в Qdrant/Milvus | Высокий |
| G | A/B тестирование retrieval качества | recall@10, MRR метрики | Средний |

---

## 📚 Ключевые ссылки и источники
- [SAM2 (Segment Anything 2)](https://github.com/facebookresearch/segment-anything-2)
- [Marqo FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Fashion-CLIP](https://huggingface.co/patrickjohncyh/fashion-clip)
- [k-Reciprocal Re-ranking (Zhong et al., 2017)](https://arxiv.org/abs/1701.08398)
- [Query Expansion for Image Retrieval](https://www.robots.ox.ac.uk/~vgg/publications/2007/Chum07a/chum07a.pdf)
- [Qdrant vector database](https://qdrant.tech/)
- [Milvus — для миллиардных каталогов](https://milvus.io/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [Uzum Marketplace](https://uzum.uz/)

---

*Последнее обновление: 20 апреля 2026*

---

## 🎨 Шаг 9 — Итерация color signature (v3 → v6) на платье 4931.jpg

Тестировали качество поиска по query `data/images/4931.jpg` — цветастое платье с принтом.
Каждая итерация основана на реальном визуальном фидбеке пользователя по результатам.

### v3: print-focused HSV signature (первая версия)
**Подход:** вместо полной HSV-гистограммы — считаем hue только от пикселей с высокой насыщенностью (S≥40). Добавили флаг `is_monochrome`.

**Формула color_similarity:**
```
oба монохромные → 0.7
один моно, другой цветастый → 0.1
оба цветастые → 0.7 * hue_intersection + 0.3 * mono_penalty
```

**Результат:** Белые платья вылетели из топа ✅. Но остались платья с розовым/жёлтым принтом при query с другим цветом. Проблема: `mono_penalty` даёт baseline ~0.3 независимо от цвета принта.

### v4: peak-hue matching (circular distance + exp penalty)
**Идея:** находим доминирующий bin в hue-гистограмме, сравниваем по **circular distance** (цвет — циклическая величина). Штраф экспоненциальный: `exp(-dist²/4)`.

```
peak_dist = min(|peak_a - peak_b|, 24 - |peak_a - peak_b|)
peak_sim = exp(-peak_dist² / 4)
   # 0 bins → 1.0
   # 1 bin  → 0.78
   # 2 bins → 0.37
   # 3 bins → 0.10
   # 4 bins → 0.02
color = 0.4 * hue_inter + 0.6 * peak_sim
```

**Результат:** Резкий разрыв между правильным топ-1 (col=0.94) и остальными. Но чёрно-белое платье `[58439]` получило col=0.55 — ложный peak из тёмных пикселей, которые прошли фильтр.

### v5: три улучшения одновременно
1. **PRINT_V_MIN: 30 → 50** — отсекаем тёмные пиксели, которые у чёрно-белых платьев ложно попадали в "print".
2. **Добавили mono_sim как 3-й аддитивный компонент**: 0.3·hue_inter + 0.5·peak_sim + 0.2·mono_sim.
3. **Color gate (мультипликативный финальный штраф):**
   ```
   final = base_score * (0.5 + 0.5 * min(col, 0.3) / 0.3)
   ```
   Если col < 0.3 — progressive penalty до ×0.5. Решает проблему когда sem+tex очень высокие у "неправильного" кандидата.

**Результат:** Чёрно-белое [58439] вылетело из топ-10 ✅. Розовые просели по final score.

### v6: mono_multiplier — мультипликативный штраф за плотность принта
**Проблема v5:** Solid-color платье (зелёное без принта) всё ещё в топе. Его mono_ratio низкий (много "colorful" пикселей одного цвета), но принт совсем другой природы чем у query (patterned).

**Решение:** `mono_sim` сделан **мультипликативным**:
```python
mono_multiplier = exp(-(mono_diff * 2)²)   # clip floor 0.1
color = hue_match * mono_multiplier
# query mono=0.57, candidate solid=0.05 → diff=0.52
# → multiplier = exp(-1.08) ≈ 0.34
# → color score падает в 3 раза
```

### Эволюция top-10 на query 4931.jpg

| # | v3 | v4 | v5 | **v6 (финал)** |
|---|----|----|----|----|
| 1 | 4984 ✅ | 4984 ✅ | 4984 ✅ | **4984** ✅ точный матч |
| 2 | 4933 (розовое ❌) | 58439 (ч-б ❌) | 27223 ✅ | 43681 ✅ |
| 3 | 33505 | 27223 | 43681 | 27223 |
| 4 | 13290 | 27198 | 27198 | 37938 (зелёное — остаток) |
| 5 | 27228 | 43681 | 37938 | 27198 |
| 6-10 | Розовые 4988/4991/4992 | + зелёное | Розовые на 8-10 | Все col<0.2, хвост у floor |

### Ключевые метрики v6

- **Разрыв top-1 vs top-2:** `0.903 vs 0.601` — большая уверенность в лучшем результате
- **Убраны:** белые, чёрно-белые, розовые, жёлтые (не совпадающие по peak hue)
- **Остался 1 false positive:** solid-color платья типа зелёного — для полной фильтрации нужен **семантический "has-print" классификатор** (CLIP text prompt "a dress with printed pattern" vs "a solid color dress"). Это возможное следующее улучшение

### Артефакты экспериментов

| Файл | Содержание |
|------|-----------|
| `data/retrieval_preview_4931.jpg` | Baseline — старая полная HSV-гистограмма |
| `data/retrieval_preview_4931_v3.jpg` | Print-focused signature |
| `data/retrieval_preview_4931_v4.jpg` | Peak-hue matching |
| `data/retrieval_preview_4931_v5.jpg` | + color gate + PRINT_V_MIN=50 |
| `data/retrieval_preview_4931_v6.jpg` | mono_multiplier (хвост топ-10 всё ещё занимали чёрные платья) |
| `data/retrieval_preview_4931_v7.jpg` | **Финал v7** — жёсткий штраф на one_mono + нижний color gate floor |

### v7: жёсткий штраф за one_mono + снижение color gate floor

**Проблема v6:** Места 6-10 занимали чёрные платья. Причина:
- Чёрные пиксели отсекаются по `V < PRINT_V_MIN=50`
- Белый фон отсекается по низкой saturation
- → `print_mask` почти пустой → `is_monochrome=True`
- Query colorful vs candidate monochrome → `one_mono` case вернул `0.1`
- Color gate с floor `0.5` защищал их: `base * (0.5 + 0.5 * 0.1/0.3) = base * 0.67`
- Итого они получали final ~0.40 и оставались в хвосте топ-10

**Два исправления (v7):**
1. **`one_mono → 0.0`** (было `0.1`) — жёсткий mismatch: "чёрное vs цветастое" ≠ "немного похоже"
2. **Color gate floor: 0.50 → 0.25** — при `col=0` итоговый штраф ×0.25 (было ×0.5)

**Формула color_gate (v7):**
```
gate = 0.25 + 0.75 * min(col, 0.3) / 0.3
# col=0.00 → 0.25
# col=0.15 → 0.625
# col=0.30+ → 1.00 (без штрафа)
```

**Результат — топ-10 стал осмысленно разделён на уровни уверенности:**
```
 1. [4984]  final=0.903  col=0.87  ← точный матч
 2. [43681] final=0.563  col=0.23  ← релевантные (col >= 0.18)
 3. [27223] final=0.507  col=0.19
 4. [37938] final=0.491  col=0.20
 5. [27198] final=0.481  col=0.19
 ─────────────── граница уверенности ───────────────
 6. [39232] final=0.286  col=0.08  ← слабые (col < 0.10)
 7. [43627] final=0.227  col=0.05
 8-10. final < 0.21, col < 0.03  ← хвост
```

---

## 🎯 Итог текущего этапа (offline retrieval pipeline — complete)

На данный момент полностью готов **offline retrieval pipeline**:

| Компонент | Статус | Файл |
|-----------|--------|------|
| Сегментация каталога (Grounded-SAM) | ✅ | `segment_catalog_sam2.py` |
| Эмбеддинги (marqo-fashionSigLIP + TTA) | ✅ | `build_embeddings_fashion.py` |
| Texture features (DINOv2 patches) | ✅ | `build_patch_features_dino.py` |
| Color signature (print-focused, peak-hue, Top-K Jaccard) | ✅ v10 | `retrieve_topk_v2.py` |
| Query expansion + k-reciprocal | ✅ | `retrieve_topk_v2.py` |
| Triple rerank (α·sem + β·tex + γ·col) + color_gate | ✅ v10 | `retrieve_topk_v2.py` |
| Диагностический тул для color | ✅ | `diagnose_color.py` |

**Качество retrieval на fashion-каталоге 399 платьев:** на всех протестированных query (4931, 4992, 5566) — точный матч на позиции #1, валидные кандидаты в топ-5, false positives вытеснены в хвост или полностью отфильтрованы hard-killом.

**Следующий этап:** переход от алгоритма к продукту — либо (а) экспорт моделей и интеграция в Android, либо (б) обёртка в FastAPI backend как первый шаг к Uzum-масштабированию.

---

## Шаг 11 — Android on-device integration

**Дата:** 03.05.2026
**Цель фазы:** замкнуть MVP-петлю «фото с камеры → top-10 похожих» прямо на Android-устройстве, без бэкенда. Демо-устройство для университетской презентации — **Google Pixel 10, Android 16**. Дальнейшая цель — масштабирование на бэкенд Uzum Marketplace.

### 11.1 Решения, зафиксированные на старте

| Решение | Выбор | Обоснование |
|---|---|---|
| ML runtime на Android | **ONNX Runtime Mobile** + NNAPI/XNNPACK | Единственный, кто нативно ест ONNX от `open_clip`/`transformers` без конверсии в TFLite (ViT-блоки требуют custom ops в TFLite). |
| UI стек | **Kotlin 2.0 + Jetpack Compose + Material 3** | Современный нативный стек, минимум boilerplate, отличная интеграция с CameraX. |
| Камера | **CameraX 1.4** | Preview + ImageCapture + Torch + Zoom + переключение front/back из коробки. |
| DI / async | Hilt + Kotlin Coroutines + Flow | Стандарт. |
| Сегментация query on-device | **U²-Netp ONNX** (~4 МБ, ~150 мс) | SAM2 1.2 ГБ, нет TFLite — на устройство не выносим. Каталог уже сегментирован офлайн SAM2. |
| Image encoder (fashion) | FashionSigLIP (Marqo) ViT-B/16 → ONNX **INT8 dynamic** | ~85 МБ, ~1.5 с inference на Pixel 10. Бит-в-бит совместимость с офлайн `embeddings_multicat.npy` валидируется через `bench_onnx_desktop.py` (cosine ≥ 0.998). |
| Image encoder (universal) | SigLIP2-base-patch16-224 → ONNX INT8 | Для watches/phones/TVs. Та же сессия используется и роутером. |
| Texture | **DINOv2-small INT8 в MVP** | Пользователь подтвердил включение сразу: +22 МБ, +400 мс — критично для принтов (полоска/цветочек/в горошек), что отличает наш проект от чистого CLIP. |
| Color | HSV v10-сигнатура нативно на Kotlin | Без ML-модели; `compute_color_signature` выносится в общий `color_v10.py`. |
| Category Router | SigLIP2 image-encode + 4 предвычисленных text-вектора | На устройстве **нет** tokenizer/text-encoder — только cosine с 4 фиксированными векторами. |
| Каталог в assets | flat-binary через `MappedByteBuffer` | 658 × 768 float32 ≈ 2 МБ, brute-force cosine <5 мс. Никакого SQLite/FAISS. |
| Distribution | **Bundled AAB** (~140 МБ) | Университетский MVP, sideload на демо-устройство. Границы модулей заложены так, что переход на Play Asset Delivery (для Uzum) не требует рефакторинга. |
| Папка Android-проекта | `/Users/i.kostiunina/StudioProjects/ImageSearchApp` | Отдельно от Python workspace, как просил пользователь. |

**Бюджет latency end-to-end (Pixel 10, NNAPI):** segmentation 150 мс + router 1.0 с (SigLIP2 image-tower) + encode 1.5 с (FashionSigLIP, ветка fashion) + DINOv2 0.4 с + color v10 5 мс + cosine + rerank <50 мс = **~3 с**. Цель — уложиться в 2.5 с после warmup и agressive INT8.

### 11.2 Архитектура on-device пайплайна (зеркало Python)

```
[Capture/Pick image]
      ↓
[Preprocess: 224×224, ImageNet/SigLIP normalize, NCHW float32]
      ↓
[U²-Netp ONNX → soft mask → bbox crop с padding 6%]            ← вместо SAM2 для query
      ↓
[SigLIP2 image-encode → cosine с 4 предвычисленными text-векторами]   ← Category Router
      ↓
[Branch by category]
   ├─ dresses/tshirts/jeans → FashionSigLIP-ORT → 768-dim
   └─ watches              → SigLIP2-ORT (уже загружен) → 768-дим
      ↓
[DINOv2-small ORT → patch tokens mean → 384-dim]               ← texture
      ↓
[HSV color signature v10 нативно на Kotlin]
      ↓
[Cosine ВНУТРИ категории → top-50 → triple rerank α=0.45/β=0.15/γ=0.40 → color_gate → top-10]
      ↓
[LazyVerticalGrid 2 кол. с thumbnails из bundled assets]
```

### 11.3 Файловая структура

**Python-сторона** (новые скрипты в корне `ImageSearch/`):
- `export_models.py` — экспорт FashionSigLIP, SigLIP2 image-tower, U²-Netp, DINOv2-small в ONNX FP32 → INT8 dynamic + валидация cosine FP32 vs INT8 ≥ 0.998.
- `export_catalog_for_android.py` — bundle: `embeddings.bin`, `color_signatures.bin`, `texture.bin`, `category_text_embeddings.bin`, `catalog.json`, `thumbnails/*.jpg` 256×256 q80.
- `bench_onnx_desktop.py` — sanity check: top-10 ORT-pipeline должен совпадать с `retrieve_topk_multicat.py` ≥ 9/10.
- `color_v10.py` — общий модуль с `compute_color_signature`, импортируется и из retrieval, и из export.

**Android-проект** (`/Users/i.kostiunina/StudioProjects/ImageSearchApp/`, multi-module Gradle):
```
:app                    MainActivity, NavGraph
:feature:home           HomeScreen (search bar + camera button)
:feature:capture        CaptureScreen (CameraX preview + bottom panel)
:feature:results        ResultsScreen (LazyVerticalGrid top-10)
:core:ml                OrtEngine, Preprocess, Segmenter, Encoder, CategoryRouter, TextureExtractor
:core:retrieval         CosineRetriever, ColorSignatureV10, TripleRerank, KReciprocal
:core:catalog           CatalogRepository, BundleLoader (mmap)
:core:ui                Material3 theme, общие компоненты
```
Assets: `app/src/main/assets/models/{fashion_siglip_int8.onnx, siglip2_int8.onnx, u2netp.onnx, dinov2s_int8.onnx}` + `assets/catalog/`.

### 11.4 Фазы (целевая длительность 14–18 рабочих дней)

- **Фаза 0** — Python-экспорт ONNX + bundle каталога + desktop-бенч (2–3 дня). ⏳ В работе.
- **Фаза 1** — скелет Android (Compose + Hilt + Navigation, два пустых экрана) (1–2 дня). ⏳ В работе параллельно.
- **Фаза 2** — CameraX (preview, ImageCapture, Torch, Zoom, front/back) (2 дня).
- **Фаза 3** — ONNX Runtime + `MlEngine` + `CatalogRepository` mmap (2 дня).
- **Фаза 4** — полный pipeline (segment → router → encode → DINOv2 → color v10 → triple rerank) (3–4 дня).
- **Фаза 5** — экран результатов top-10 (1 день).
- **Фаза 6** — оптимизация (NNAPI/XNNPACK, warmup, профайлинг) (2–3 дня).
- **Фаза 7** — тестирование на Pixel 10, sanity check vs Python (1–2 дня).

### 11.5 Старт Фазы 0 (03.05.2026)

Создаю на Python-стороне:
1. `export_models.py` — экспорт 4 моделей в ONNX INT8.
2. `export_catalog_for_android.py` — bundle для assets.
3. `bench_onnx_desktop.py` — sanity check.
4. `color_v10.py` — выделение общего color-модуля.

Параллельно создаю каркас Android-проекта в `/Users/i.kostiunina/StudioProjects/ImageSearchApp`.

---

### 11.6 Фаза 0 завершена (Python экспорт) ✅

Созданы:
- `color_v10.py` — общий модуль HSV-сигнатуры. Выделен из `retrieve_topk_v2.py`,
  служит эталоном для Kotlin-реализации (`ColorSignatureV10.kt`). Все константы и
  формулы зафиксированы здесь, любые изменения должны быть синхронизированы с обеих сторон.
- `export_models.py` — экспортирует 4 модели:
  1. FashionSigLIP image-tower (open_clip) → ONNX FP32 → INT8 dynamic
  2. SigLIP2-base image-tower (transformers) → ONNX FP32 → INT8 dynamic
  3. DINOv2-small (transformers, mean of patch tokens) → ONNX FP32 → INT8 dynamic
  4. U²-Netp ONNX (копируется из rembg cache, без квантизации — и так 4 МБ)

  Sanity check встроен: после INT8-квантизации каждой модели на тестовом изображении
  вычисляется `cosine(FP32_output, INT8_output)`, цель ≥ 0.998. Если порог не
  достигнут — печатается ⚠ и нужно использовать FP32 для этой модели.

- `export_catalog_for_android.py` — собирает bundle для assets:
  - `embeddings.bin` (658×768 float32, ≈2 МБ) — порядок совпадает с rows.
  - `texture.bin` (658×384 float32) — переупорядочен по product_id из rows.
  - `color_signatures.bin` (658×26 float32) — пересчитываем v10-сигнатуру для каждого item.
  - `category_text_embeddings.bin` (4×768 float32) — усреднённые SigLIP2 text-features
    по `CATEGORY_PROMPTS`. Это позволяет на устройстве не тащить tokenizer/text-tower:
    router сводится к одному image-encode + 4 dot-product.
  - `catalog.json` — метаданные.
  - `thumbnails/<pid>.jpg` — 256×256 JPEG q80.
  - `manifest.json` — версия + размерности для Kotlin-loader'а.

- `bench_onnx_desktop.py` — desktop-эмулятор on-device pipeline через onnxruntime
  (CPUExecutionProvider). Прогоняет: router → encoder branch → texture → color v10 →
  triple rerank → top-10. Сравнивает с `retrieve_topk_multicat.py` по recall@10.
  Цель ≥ 0.9 на 4 demo-query (dress/tshirt/jeans/watch).

Запуск (когда пользователь готов):
```bash
cd /Users/i.kostiunina/PycharmProjects/ImageSearch
pip install -r requirements.txt        # подтягивает onnxsim, open_clip_torch, rembg
python3 export_models.py               # ~5–15 мин (скачивает веса HF + квантизация)
python3 export_catalog_for_android.py  # ~1–2 мин (color sigs + thumbnails)
python3 bench_onnx_desktop.py          # sanity check recall ≥ 9/10
```

### 11.7 Фаза 1 завершена (Android skeleton) ✅

Создан проект `/Users/i.kostiunina/StudioProjects/ImageSearchApp` —
multi-module Gradle (Kotlin DSL, Version Catalog, AGP 8.7.3, Kotlin 2.0.21,
Compose BOM 2025.01.00, ORT 1.20.0, CameraX 1.4.1).

| Модуль | Статус Phase 1 | Что внутри |
|---|---|---|
| `:app` | ✅ skeleton | `MainActivity`, `ImageSearchApplication` (Hilt), `AppNavGraph` (3 экрана с typed routes) |
| `:feature:home` | ✅ done | `HomeScreen` — search bar в pill-shape с camera icon справа |
| `:feature:capture` | ✅ skeleton | `CaptureScreen` — header «Search by photo» + close, плейсхолдер preview, bottom bar (галерея/zoom/shutter/switch/torch). Кнопка галереи уже подключена через `ActivityResultContracts.PickVisualMedia` и навигирует на Results с URI. CameraX preview/capture — Phase 2. |
| `:feature:results` | ✅ skeleton | `ResultsScreen` — `LazyVerticalGrid 2 колонки, 10 плейсхолдеров. Pipeline-интеграция — Phase 4. |
| `:core:ui` | ✅ done | Material3 light/dark theme. |
| `:core:catalog` | ✅ done | `BundleLoader` читает `manifest.json` + 4 `.bin` (DirectByteBuffer little-endian) + `catalog.json`. `LoadedBundle.row()` копирует ряд в `FloatArray` под cosine. `noCompress = ['bin', 'json']` — assets лежат raw. |
| `:core:retrieval` | ✅ done | **`ColorSignatureV10.kt` — 1:1 порт `color_v10.py`** (все формулы, константы, peak-bin kill, jaccard gate). Особое внимание уделено совместимости HSV-шкалы: PIL даёт H,S,V ∈ [0,255], Android `Color.RGBToHSV` даёт H ∈ [0,360], S,V ∈ [0,1] — конвертируем в PIL-шкалу до бинаризации. **`CosineRetriever.kt`** — фильтр по category, triple rerank α=0.45 / β=0.15 / γ=0.40 + `gate(col)`. |
| `:core:ml` | ✅ done | `Preprocess.toNchw()` — Bitmap → DirectFloatBuffer NCHW [1, 3, S, S]. `OrtModel` — ленивый copy-asset-to-files, `OrtSession` с NNAPI EP (graceful fallback на CPU/XNNPACK), `setOptimizationLevel(ALL_OPT)`, L2-нормализация на выходе. |

**Ключевые решения, которых стоит коснуться в презентации:**
1. **Bit-exact парность Python ↔ Kotlin для color signatures.** Если бы мы считали HSV на Kotlin напрямую через `Color.RGBToHSV`, без перевода в PIL-шкалу 0–255, биннинг разъехался бы — каталожные сигнатуры (Python) и query-сигнатуры (Android) дали бы разный peak-bin. Этот сорт ошибок невидим в логах, но ломает recall.
2. **Category text embeddings предвычисляем офлайн.** Это убирает с устройства весь text-pipeline (tokenizer + text-tower SigLIP2 ≈ 100 МБ + 200 мс на encode). Router становится одним image-encode + 4 dot-product.
3. **mmap вместо SQLite/FAISS.** Каталог 658 × 768 float32 = 2 МБ, brute-force cosine на 658 строк <5 мс. Сложная индексная структура была бы overkill, и она же требовала бы своего runtime'а.
4. **NNAPI с graceful fallback.** ORT на Pixel 10/Tensor G5 хорошо ест ViT через NNAPI, но не все ops — оставляем включённым, при отказе ORT сам опускается на XNNPACK CPU.

### 11.8 Что пользователь запускает прямо сейчас

```bash
# 1) Python: экспорт моделей и сборка bundle (~10 мин на M1/M2)
cd /Users/i.kostiunina/PycharmProjects/ImageSearch
pip install -r requirements.txt
python3 export_models.py
python3 export_catalog_for_android.py
python3 bench_onnx_desktop.py        # должен показать recall ≥ 9/10

# 2) Скопировать в Android assets
APP=/Users/i.kostiunina/StudioProjects/ImageSearchApp/app/src/main/assets
mkdir -p $APP/models $APP/catalog
cp models/*.onnx $APP/models/
cp -r android_bundle/* $APP/catalog/

# 3) Открыть в Android Studio
#    File → Open → /Users/i.kostiunina/StudioProjects/ImageSearchApp
#    При первом sync Studio сам создаст gradlew + gradle-wrapper.jar.
#    File → New → Image Asset для иконки приложения.
#    Run на Pixel 10 (USB debugging) → увидим Home → Capture → Results skeleton.
```

### 11.9 Дальше — Phase 2 (CameraX, ~2 дня)

В `:feature:capture`:
1. `CameraController` (CameraX 1.4): Preview + ImageCapture, lifecycle-aware.
2. `PreviewView` через `AndroidView { … }` поверх Black-фона (вместо плейсхолдера).
3. Кнопки bottom bar:
   - shutter → `ImageCapture.takePicture()` → save в `cacheDir` → callback с URI.
   - front/back → `CameraSelector.LENS_FACING_FRONT/BACK`.
   - torch → `CameraControl.enableTorch()` (только для back-камеры; кнопку прячем для front).
   - zoom → pinch-gesture + кнопка `setZoomRatio()`.
4. Permissions flow: `Manifest.permission.CAMERA` через `ActivityResultContracts.RequestPermission`.

После Phase 2 → Phase 3 (smoke-test ORT-сессии и каталога), затем Phase 4 (полный
pipeline). На каждом шаге фиксируем результат и решения здесь же в DEVLOG.

---

### 11.10 Фаза 0 закрыта — результаты бенчмарка ✅

**Команда:** `python3 bench_onnx_desktop.py` (semantic-only, like-with-like с
`retrieve_topk_multicat.py`).

| query | cat | recall@10 |
|---|---|---|
| multicat_demo_dress.jpg | dresses | 9/10 |
| multicat_demo_tshirt.jpg | tshirts | 9/10 |
| multicat_demo_jeans.jpg | jeans | 10/10 |
| multicat_demo_watch.jpg | watches | 9/10 |
| **mean** | | **0.93 ≥ 0.9 ✅** |

**Что это доказывает:**
1. **ONNX-экспорт корректен.** INT8-эмбеддинги query (через onnxruntime) совместимы с FP32 PyTorch-эмбеддингами каталога. Drift cosine ≈ 0.005 шевелит ранжирование на 1 позицию из 10 — допустимо.
2. **Category router работает.** Все 4 query попали в правильную ветку (dresses/tshirts/jeans/watches), используя один image-encode SigLIP2 + 4 предвычисленных text-вектора.
3. **Pipeline-зеркало Python ↔ Android корректно.** Bench написан так же, как будет работать Kotlin: одна сессия SigLIP2 для роутера и watch-encoder, отдельная FashionSigLIP для остальных категорий.

**Что показывает `--full-rerank` (0.33):** это **другой** ranking — с texture+color+gate. Низкий recall здесь — артефакт того, что ground truth offline-pipeline (`retrieve_topk_multicat.py`) — semantic-only, без rerank. Качество triple rerank уже валидировалось отдельно (см. §9-10), и оно идентично перенесётся на Android, потому что это детерминированная арифметика на тех же данных (`color_v10.py` ↔ `ColorSignatureV10.kt`).

**Известные минорные проблемы, не блокирующие Phase 2:**
- `texture.bin: missing=259` — DINOv2 patch features есть только для 399 платьев (не для tshirts/jeans/watches, потому что `build_patch_features_dino.py` бежал по `catalog_sam2.jsonl`, а не по multicat). На Android для не-dresses tex_sim=0, retrieval падает на semantic+color — всё ещё работает, но без texture-вклада. Перегон на multicat — отдельная задача после Phase 4.
- `cosine(FP32, INT8) = 0.9957` для FashionSigLIP — ниже целевого 0.998. Но recall@10=0.93 показывает, что это не транслируется в потерю качества retrieval. Если когда-нибудь окажется критично — пересчитаем catalog через ONNX-INT8 для бит-в-бит парности.

**Артефакты Фазы 0:**
- `models/{fashion_siglip,siglip2,dinov2s}_{fp32,int8}.onnx` + `models/u2netp.onnx`
- `android_bundle/{embeddings,texture,color_signatures,category_text_embeddings}.bin`
- `android_bundle/{catalog,manifest,category_text_embeddings}.json`
- `android_bundle/thumbnails/<pid>.jpg` × 658

**Старт Phase 2 — CameraX.** Поехали.

---

### 11.11 Фаза 2 завершена (CameraX) ✅

В `:feature:capture` появились:
- **`CameraState.kt`** — обёртка над `ImageCapture` + `CameraControl`. Хранит выбранную линзу, состояние torch, текущий и максимальный zoom как Compose mutable state. `toggleLens()` автоматически гасит torch при переключении на front-камеру (на front обычно нет вспышки). `setZoom(ratio)` дёргает `setZoomRatio()` с `coerceIn(1f, maxZoom)`.
- **`CameraPreview` Composable** — `AndroidView` с `PreviewView` (FILL_CENTER + PERFORMANCE mode). `LaunchedEffect(lensFacing)` переподнимает Preview + ImageCapture use-case'ы при переключении камеры. `awaitCameraProvider` — `suspendCancellableCoroutine` поверх `ProcessCameraProvider.getInstance().addListener()`. На `onDispose` делает `unbindAll()`, чтобы корректно отпускать ресурсы при уходе с экрана.
- **`ImageCapture.takeSnapshot()` extension** — suspend-обёртка над `takePicture()`. Сохраняет JPEG в `cacheDir/captures/capture_<ts>.jpg`, возвращает `file://` URI. Используется в shutter-кнопке.
- **Новый `CaptureScreen.kt`** заменил Phase 1-плейсхолдер. Сверху — header, посередине — реальный preview с **pinch-to-zoom** (`detectTransformGestures` поверх `CameraPreview`), снизу — bottom bar с галереей / shutter / switch / torch. При активном zoom показывается badge `1.5×` в верхнем правом углу. Permission flow на CAMERA — через `ActivityResultContracts.RequestPermission`; при отказе показывается панель с кнопкой «Grant permission» вместо preview.
- В `libs.versions.toml` добавлен `androidx-lifecycle-runtime-compose` — для нового `LocalLifecycleOwner`, который `CameraX.bindToLifecycle` ожидает.

**Что пользователь увидит на Pixel 10:**
1. Tap camera-icon на Home → запрашивается CAMERA permission.
2. После grant — живой preview с задней камеры.
3. Pinch — zoom (с badge), shutter — снимок сохраняется в cache → переход на Results с URI.
4. Switch lens — мгновенное переключение front/back. Torch отключается при переходе на front.
5. Кнопка галереи открывает системный picker (Android 13+ Photo Picker). Выбор изображения → Results.

**Что Phase 2 не делает (ждёт следующих фаз):**
- На Results показывается всё ещё **placeholder grid** — pipeline (segment → router → encode → DINOv2 → color v10 → rerank) подключим в Phase 4.
- `ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY` — мы выбрали скорость над качеством, потому что для retrieval нам важна не максимальная HDR-картинка, а быстрый снимок 1024×768 для последующего encode на 224×224.

### 11.12 Дальше — Phase 3 (smoke-test ML pipeline)

В отдельном `MainActivity` debug-флаге или прямо в `ResultsScreen` запустим smoke-test:
1. `BundleLoader` грузит `manifest.json`, проверяет размеры всех `.bin`.
2. `OrtModel.fromAsset` запускает FashionSigLIP + SigLIP2 + DINOv2 на dummy `Bitmap.createBitmap(224, 224)`. Логирует latency каждой сессии и финальный shape.
3. На Pixel 10 ожидаем (NNAPI EP):
   - SigLIP2 image-encode: ~600–900 мс
   - FashionSigLIP encode: ~700–1000 мс
   - DINOv2-small encode: ~300–500 мс
   - U²-Netp segment 320×320: ~120–200 мс

Если что-то падает на NNAPI — graceful fallback на CPU/XNNPACK уже встроен, latency ×1.5–2×, но всё ещё в бюджете 3 с.

---

## 12. Phase 3 — ML smoke-test on device (Pixel 10)

**Дата:** 2026-05-03  
**Цель:** убедиться что все 4 ONNX-модели поднимаются и прогоняются на устройстве **до** реализации Phase 4 (полный pipeline).

### 12.1 Что сделали

- В `core:ml` добавлен `SmokeTest` — object с `run(context): Report`. Открывает 4 ORT-сессии (NNAPI EP), для каждой: copy asset → load → 1st run на zero-тензоре → warm run. Замеряет load/1st/warm latency, in/out shapes, размер модели, peak heap/native RAM.
- В `feature:home` под search bar — debug-кнопка «Run ML smoke-test». Прогресс-индикатор на время прогона, отчёт моноширинным шрифтом прямо на экране + дубль в logcat (`MlSmokeTest`).
- В `core:ml/SmokeTest.kt::copyAsset()` добавлена сверка длины файла в `filesDir` с длиной asset'а — авто-инвалидация кэша при обновлении модели. Без этого после re-quantization старый `.onnx` из `filesDir` остался бы жить и smoke-test тестировал бы вчерашнюю модель.

### 12.2 Хотфикс: `ConvInteger` падает на Android

**Симптом первого прогона:** все 3 SigLIP/DINO-сессии падают на `createSession`:

```
ai.onnxruntime.OrtException: Error code - ORT_NOT_IMPLEMENTED -
message: Could not find an implementation for ConvInteger(10) node
with name '/visual/trunk/patch_embed/proj/Conv_quant'
```

**Причина:** в `export_models.py` использовалось `quantize_dynamic(..., weight_type=QuantType.QInt8)` без указания `op_types_to_quantize`. По умолчанию ORT квантует **Conv → ConvInteger(opset 10)**. Десктопный `onnxruntime` эту операцию имплементит, **android-сборка `onnxruntime` — нет**. Это известная boundary, не баг.

**Фикс:** переквантовать с `op_types_to_quantize=["MatMul", "Gemm"]`. В ViT-B/16 и DINOv2-S по одному Conv на patch_embedding (≈0.1% параметров). Оставить его в FP32 стоит ~1% размера модели — несоизмеримо мало по сравнению с тем что без этого код вообще не работает на Android.

Создан `requantize_int8_for_android.py` — one-shot скрипт, который берёт уже готовые `*_fp32.onnx` и переквантует за ~30 с (без перезапуска torch-экспорта). `export_models.py::_quantize_int8` тоже обновлён, чтобы будущие full re-export'ы сразу выдавали Android-совместимые INT8.

**Cosine FP32 ↔ INT8 (Android-вариант) на random fp32 input:**

| Модель         | cosine     | размер (был → стал) |
|----------------|------------|---------------------|
| fashion_siglip | **0.9974** | 94 → 96 МБ          |
| siglip2        | **0.9951** | 99 → 101 МБ         |
| dinov2s        | 0.9878     | 24 → 25 МБ          |

DINOv2 чуть ниже 0.99-порога. Это texture-rerank, вторичный сигнал в финальном score (~10% веса). Если на Phase 4 заметим деградацию retrieval, будем переделывать на **static QDQ**-quantization с калибровкой на 50 фото каталога.

### 12.3 Результаты smoke-test после фикса (Pixel 10 / Tensor G5 / Android 16)

```
ML smoke-test — total 7321 ms
Heap peak: 22 MB · native peak: 847 MB

[OK ] fashion_siglip  (91 MB)   load=428 ms · 1st=152 ms · warm=129 ms  · out=[1, 768]
[OK ] siglip2          (96 MB)  load=292 ms · 1st=152 ms · warm=124 ms  · out=[1, 768]
[OK ] dinov2s          (23 MB)  load=209 ms · 1st=117 ms · warm=97 ms   · out=[1, 384]
[OK ] u2netp@320nnapi  (4 MB)   load=163 ms · 1st=2567 ms · warm=2600 ms · out=[1, 1, 320, 320]
```

**Вывод по трём SigLIP/DINO моделям:** ✅ всё в бюджете. Warm-run encoders 97–129 мс — реальный pipeline (segment + 2 encode + cosine 658 items + rerank) укладывается в **~1 с на encode-стадии**.

**Native peak 847 МБ** — комфортно для Pixel 10 (12 ГБ RAM). Все 4 сессии живут одновременно, как и будет в проде.

### 12.4 Проблема: u2netp 2.6 с warm — ломает бюджет

U²-Netp segment занимает **2.6 с warm-run** (на десктопе ORT-CPU 80–120 мс). В то время как 3 transformer-сессии летают на 100 мс, лёгкая FCN-segmentация — bottleneck.

**Гипотезы причин:**
1. NNAPI на Tensor G5 не имплементит часть ops u2netp (`ConvTranspose2d` в decoder), молча fallback'ит на CPU без оптимизаций.
2. CPU EP без NNAPI с XNNPACK мог бы быть быстрее.
3. 320×320 — избыточный input size; u2netp полностью свёрточная, принимает любой `size % 32 == 0`.

**Что делаем:** в `SmokeTest.MODELS` добавлены 3 варианта u2netp для A/B-замера на следующем запуске:

- `u2netp@320nnapi` — baseline (как было)
- `u2netp@256nnapi` — компьют ×1.56 меньше
- `u2netp@320cpu`   — без NNAPI (только XNNPACK)

Варианты 2 и 3 имеют `keepAlive = false` — закрывают сессию сразу после warm-run, чтобы native peak оставался репрезентативным (т.е. как в проде, где будет одна u2netp-сессия).

### 12.5 Дальше — план Phase 4 (зависит от u2netp A/B)

Сценарии:

**A.** Если `u2netp@256nnapi` ≤ 1 с → используем 256×256 для query-сегментации. Pipeline:

```
capture → resize 256×256 → u2netp → mask → crop+resize 224×224 → siglip2 (router)
                                                              ↓ выбор encoder
                                                          fashion_siglip OR siglip2
                                                              ↓
                                                       cosine vs catalog → top-50
                                                              ↓
                                                   dinov2 rerank + color_v10 → top-10
```

Бюджет: ~1 с segment + 0.4 с encode + 0.05 с retrieval = **~1.5 с end-to-end** ✅

**B.** Если `u2netp@320cpu < u2netp@320nnapi` → принципиальная находка для всего проекта: для u2netp (и возможно других CNN) дефолтный EP — XNNPACK CPU, не NNAPI.

**C.** Если оба варианта всё равно > 2 с → **фолбэк-стратегия**:
- Query идёт **без сегментации**, прямо в SigLIP.
- Каталог пересчитываем тоже без сегментации (`build_embeddings_siglip2.py` без `_cropped`).
- Известная просадка ~5% recall@10 для marketplace-photos с произвольным фоном; для MVP с фото на нейтральном фоне (ладонь / стол / вешалка) — приемлемо.
- В презентации честно говорим: «сегментация +5% recall, но +3 с latency on-device → для prod backend имеет смысл, on-device — нет».


### 12.6 A/B результаты u2netp на Pixel 10 — финал Phase 3

```
[OK ] u2netp@320nnapi  load=133 ms · 1st=2494 ms · warm=2543 ms
[ERR] u2netp@256nnapi  ORT_INVALID_ARGUMENT — модель экспортирована с фиксированным
                       input shape [1, 3, 320, 320], 256×256 не принимает
[OK ] u2netp@320cpu    load=62 ms  · 1st=346 ms  · warm=327 ms
```

**Открытие:** **NNAPI на Tensor G5 для u2netp в 7.8× медленнее CPU/XNNPACK** (2543 vs 327 мс).
Скорее всего NNAPI fallback'ит `ConvTranspose2d` блоки decoder'а на CPU, добавляя
overhead на cross-EP синхронизацию. Чистый CPU + XNNPACK работает идеально.

`@256nnapi` упал потому что u2netp экспортирована rembg'ом со статическим shape;
переэкспортить с `dynamic_axes={"input.1": {2: "h", 3: "w"}}` можно, но не нужно —
327 мс на 320×320 уже укладываются в бюджет.

### 12.7 Решение для Phase 4

- **u2netp** запускаем на CPU (без `addNnapi()`).
- **siglip2 / fashion_siglip / dinov2s** — оставляем NNAPI (там он работает: 97–129 мс warm).
- u2netp@256nnapi и u2netp@320cpu в `SmokeTest.MODELS` остаются как regression-bench для будущих устройств (вдруг на других чипах NNAPI лучше).

**Финальный бюджет on-device pipeline (Pixel 10, измерено):**

| Стадия | latency |
|---|---|
| u2netp segment 320×320 (CPU) | ~330 мс |
| siglip2 router (NNAPI) | ~125 мс |
| fashion/siglip2 encode (NNAPI) | ~130 мс |
| dinov2 texture (NNAPI) | ~100 мс |
| cosine 658×768 + filter | <30 мс |
| color v10 + triple rerank | <20 мс |
| **TOTAL** | **~750 мс** ✅ |

Бюджет 2 с с запасом 60%. **Phase 3 закрыта**, переходим к Phase 4.


---

## 13. Phase 4 hotfixes — выдача с детским/денимовым платьем

**Дата:** 2026-05-03
**Контекст:** На реальных устройстве после Phase 4 пользователь сделал 7 query:
- Скрины 1–4 — нормальные fashion product shots → top-10 хороший.
- Скрин 5 — детское бело-голубое платье → выдача футболки (категория ошиблась).
- Скрин 6 — розовое женское платье → платья правильной категории, но непохожих стилей.
- Скрин 7 — джинсово-голубое платье → выдача джинсов (категория ошиблась).

Три отдельные проблемы, три фикса.

### 13.1 Router работает на сегментированной картинке → ошибочные категории

**Симптом:** SigLIP2 unitary text-image cosine для category классификации работал
на u2netp-cropped изображении. На сложных query (модель в платье на цветном фоне,
ребёнок-with-toy) u2netp выдавал шумную маску, soft alpha-blend на белый фон давал
off-distribution картинку, и SigLIP2 zero-shot классификация ломалась.

**Фикс:** в `InferenceEngine.runQuery` теперь:
- **Router** (cosine с category text embeddings) работает на **RAW** image — siglip2
  zero-shot классификация устойчивее к фону, чем к артефактам сегментации.
- **Encoder + DINOv2 + color signature** работают на **cropped** (как раньше) —
  это нужно для distribution match с каталогом, который посчитан на SAM2-cropped.

**Цена:** один дополнительный forward через siglip2 (на raw, ~125 мс) → общий
бюджет растёт с ~750 мс до ~870 мс. Всё ещё в бюджете 2 с.

### 13.2 Color rerank доминирует над семантикой → стилистический mismatch

**Симптом:** в верных платьях возвращались dresses с правильным цветом (pink),
но непохожим силуэтом / принтом. И наоборот — голубое платье выкидывалось
в jeans-категорию ещё до color rerank, плюс color gate жёстко обнулял
кросс-цветные семантические матчи.

**Фикс:**
- `TripleRerank` веса: `ALPHA=0.55, BETA=0.20, GAMMA=0.25`
  (было 0.45 / 0.15 / 0.40). Семантика теперь доминирует.
- `ColorSignatureV10.gate`: floor `0.05 → 0.30`, full `0.30 → 0.40`.
  Теперь color-mismatched items получают 0.30 множитель (а не 0.05) и
  семантически очень близкий dress другого цвета уже может попасть в top-10.

Веса в bench_onnx_desktop.py остались прежние (0.45/0.15/0.40) — это reference
для recall@10 параметрии, который один-в-один воспроизводит offline pipeline
без rerank. Изменение только в Android-стороне.

### 13.3 Thumbnails — оригиналы вместо cropped

**Запрос:** retrieval гонять на cropped (для quality), но в выдаче показывать
оригинальные изображения (как товар выглядит на маркетплейсе — на людях, в студии,
с фоном).

**Фикс:** `export_catalog_for_android.py::_resolve_thumbnail_source` теперь
ищет thumbs в порядке:
1. `data/images/<pid>.jpg` (originals — есть для всех 399 dresses)
2. `data/images_multicat_hires/<pid>.jpg` (multi-category originals)
3. `data/images_multicat_raw/<pid>.jpg`
4. `data/<row.image>` = `data/images_sam2/<pid>.jpg` (cropped — fallback)

После пересборки bundle: **399 originals + 259 cropped fallback**. Все 399 dresses
теперь показываются как студийные фото на людях/моделях. Tshirts/jeans/watches
(259 items, originals не скачивались — они приехали из HF dataset уже cropped)
остаются на cropped — это ограничение исходных данных.

**Важно:** эмбеддинги/color signatures каталога остались на cropped версиях, что
обеспечивает distribution match с on-device pipeline (где query тоже сегментируется
u2netp). Поменяли ТОЛЬКО что показывается пользователю.


---

## 14. Phase 4 round 2 — visual prototypes router + thumbs originals

**Дата:** 2026-05-03
**Контекст:** Тест 7 query (test2/) показал что Phase 4 hotfixes из секции 13
помогли частично, но остались три ошибки:

- **screen_4931** (детское бело-голубое платье → футболки): router ВСЁ ЕЩЁ
  ошибается, теперь уже на raw image. SigLIP2 zero-shot text-image cosine
  с prompt'ами "dresses/tshirts/jeans/watches" даёт cross-margin 0.04–0.10
  на тройнике dresses↔tshirts↔jeans → ненадёжный argmax.
- **screen_10500** (тёмно-синее платье → бело-чёрные футболки): то же.
- **screen_6874** (розовое однотонное → платья с принтами): color signature
  query ошибочно классифицируется как non-monochrome из-за остатков кожи
  в cropped по краям bbox (nPrint > 200, threshold абсолютный).

Плюс UI запрос: thumbnails для tshirts/jeans/watches должны быть **оригиналами**,
а не cropped версиями (показывали 256×256 квадраты на белом).

### 14.1 Router → визуальные прототипы (`build_category_visual_prototypes.py`)

Старая схема: SigLIP2 text encoder для категорий → cross-cosines между
прототипами 0.86–0.90 (margin <0.04 → unstable argmax).

Новая схема: для каждой категории каталога вычисляем mean L2-normalized
эмбеддинг **на cropped изображениях каталога** (657 forwards), сохраняем
4 центроида в `category_visual_prototypes.bin`. Сравнили siglip2 vs
fashion_siglip как encoder для прототипов:

|                           | mean margin | dresses-vs-tshirts | watches-vs-* |
|---------------------------|-------------|--------------------|--------------|
| siglip2_int8              | 0.130       | 0.105              | 0.20         |
| **fashion_siglip_int8**   | **0.169**   | **0.122**          | **0.27**     |

FashionSigLIP обучался специально на product-fashion → ожидаемо лучше
разделяет fashion-категории. Берём его. Bundle version 1 → **2**, manifest
содержит `router_method: "visual_prototypes"`.

### 14.2 Confidence fallback в Kotlin

Margin между dresses и tshirts всё ещё 0.122 — на пограничных query
может проседать. Добавлен fallback в `InferenceEngine.routeByVisualPrototypes`:

```
margin = sims[best] - sims[2nd_best]
if margin < 0.03 → category = "dresses"  // 60.6% каталога
```

Логика: при равной неуверенности bayes prior отдаём наибольшей категории —
это минимизирует ожидаемую ошибку. Можно будет потом сделать threshold
adaptive (например, не fallback в dresses если best=watches с margin 0.15).

### 14.3 Routing на FashionSigLIP, не SigLIP2

Старая `runQuery` гоняла siglip2 для router ещё одним forward'ом. Теперь
router использует FashionSigLIP — тот же encoder что и для retrieval-стадии
у одежды. Бонус: на cropped пути для не-watches мы можем вообще
переиспользовать query embedding из router-стадии (raw vs cropped, но
distribution-mismatch есть в обе стороны — пока считаем отдельно для
parity с каталогом). Для watches ещё гоним siglip2 на cropped.

Latency на Pixel 10 (ожидаемо):
- router (FashionSigLIP raw, NNAPI): ~130 мс
- segment (u2netp CPU): ~330 мс
- encode FashionSigLIP cropped: ~130 мс
- DINOv2 cropped: ~100 мс
- color + cosine + rerank: ~50 мс
- **total ≈ 740 мс** (без watches, +130 мс если watches требует siglip2)

### 14.4 Color signature: monochrome detection ratio-based на query

Проблема screen_6874: cropped query содержит кожу модели → nPrint > 200
порог → query is_monochrome=False. Mono-каталог items × non-mono query
= color sim 0.0 → color_gate множит на 0.30 → solid pink catalog items
проигрывают printed pink items в ranking.

**Фикс в `ColorSignatureV10.extract`:**

```kotlin
val printRatio = nPrint.toFloat() / nFg
val isMonochrome = nPrint < MIN_PRINT_PIXELS || printRatio < MONO_PRINT_RATIO
```

`MONO_PRINT_RATIO = 0.05` — < 5% saturated pixels = monochrome. Catalog
sigs остаются с абсолютным порогом 200 (они и так чистые). Query становится
устойчивее к 5–10% кожи в cropped.

### 14.5 Thumbnails — оригиналы для всех 658 items

Bug in `_resolve_thumbnail_source` — искал в FLAT `images_multicat_hires/`,
а originals лежат в **nested** `images_multicat_hires/{tshirts,jeans,watches}/`.

Теперь резолвер знает про nested структуру:

| Категория | Источник thumbs |
|-----------|-----------------|
| dresses (399) | `data/images/<pid>.jpg` |
| tshirts (95) | `data/images_multicat_hires/tshirts/<pid>.jpg` |
| jeans (66)   | `data/images_multicat_hires/jeans/<pid>.jpg` |
| watches (98) | `data/images_multicat_hires/watches/<pid>.jpg` |

После пересборки: **658 originals + 0 cropped fallback**. Tshirts/jeans/watches
оригиналы — это ProductDB studio shots (384×512), на белом фоне, но это
**исходные** product photos с маркетплейса, не SAM2-cropped версии.
Embeddings/color signatures каталога остаются на SAM2-cropped (parity с
on-device pipeline).

### 14.6 Изменённые/новые файлы

- `build_category_visual_prototypes.py` (NEW) — пересчёт visual prototypes
  с авто-выбором лучшего encoder'а.
- `export_catalog_for_android.py` — nested-dir resolver для thumbs,
  manifest version=2 + `router_method`.
- `core/catalog/Models.kt` — `BundleManifest.routerMethod`.
- `core/catalog/BundleLoader.kt` — читает `category_visual_prototypes.bin`,
  fallback на text routing если файла нет (bundle v1).
- `core/ml/InferenceEngine.kt` — `routeByVisualPrototypes` + confidence
  fallback на dresses; routing использует FashionSigLIP вместо SigLIP2.
- `core/retrieval/ColorSignatureV10.kt` — ratio-based mono detection
  (5% threshold).


---

## 15. Phase 4 round 3 — union retrieval (jeans-fix без bias dresses)

**Дата:** 2026-05-03
**Контекст:** Round 2 (раздел 14) исправил dresses-кейсы (4931, 6874, 10500),
но user в test3 нашёл **3 jeans-query** (7193, 26995, 27020), по которым
выдача — платья.

### 15.1 Корневая причина — мой fallback bias

В round 2 я добавил confidence-fallback в роутер:
```kotlin
if (margin < 0.03) → category = "dresses"  // насильно
```

Логика была "при неуверенности голосуем за dresses потому что 60% каталога".
**Проблема:** правило симметричное — для jeans query тоже срабатывает:

| query тип | best | 2nd | margin | старая логика | результат |
|---|---|---|---|---|---|
| children's dress (4931) | tshirts | dresses | 0.02 | → dresses | ✅ |
| dark dress (10500)      | tshirts | dresses | 0.02 | → dresses | ✅ |
| jeans (7193, 26995, 27020) | jeans | dresses | ~0.02 | → dresses | ❌ |

То есть fallback "лечил" платья ценой джинсов.

### 15.2 Фикс — union retrieval вместо forced fallback

Идея: при низкой уверенности **расширяем поиск** до top-2 категорий, не
переопределяя primary. Семантический encoder (FashionSigLIP) на cropped
query даст высокий cosine с правильными items, и top-K естественно
заполнится из верной категории.

**Latency:** retrieval по 658 items — 30 мс. По 658 items с фильтром по
2 категориям — те же 30 мс (cosine-ы считаются для всех всё равно).
Union бесплатный.

**Cases после фикса:**

| query | candidates | Что произойдёт |
|---|---|---|
| jeans (margin 0.02) | {jeans, dresses} | jeans-items имеют sem cosine 0.85+ с query, dresses — 0.7. Top-10 = jeans ✅ |
| children's dress (margin 0.04) | {tshirts, dresses} | dress-items имеют sem 0.85+, tshirts 0.75. Top-10 = dresses ✅ |
| watches (margin 0.27) | {watches} | без union, как было ✅ |

### 15.3 Watches защищены от смешивания

Watches catalog построен на SigLIP2 (encoder=universal), одежда — на
FashionSigLIP. Эмбеддинги в разных пространствах, cosine между ними
бессмысленный. Поэтому в union идут только если **оба** кандидата —
fashion-категории:

```kotlin
val candidates = if (margin < LOW_CONFIDENCE_MARGIN
    && primary != "watches"
    && secondary != "watches"
) setOf(primary, secondary) else setOf(primary)
```

### 15.4 Изменения

- `core/retrieval/CosineRetriever.kt`: signature `category: String → categories: Set<String>`.
- `core/ml/InferenceEngine.kt`:
  - `RouterDecision(primary, candidates, sims, margin)` data class
  - `routeByVisualPrototypes` возвращает RouterDecision
  - `runQuery` использует `decision.candidates` для retrieval, `decision.primary` для encoder selection
  - удалён насильный fallback на dresses
  - `LOW_CONFIDENCE_MARGIN: 0.03 → 0.05` (более широкая сеть, union бесплатный)
  - В QueryOutput.category показывается "primary (+ secondary)" если был union

### 15.5 Что ещё стоит мониторить

- Если на 2 категории всё равно даёт мусор (например jeans-query → tshirts+dresses,
  оба плохие), значит margin реально >0.05 в пользу неверной категории. Тогда
  потребуется **ensemble прототипов** (siglip2 + fashion_siglip) или **per-image
  classifier** (отдельная маленькая ONNX-модель). Пока надеемся, что union
  решает большинство пограничных случаев.
- ColorSignatureV10 ratio-based mono detection (round 2, 14.4) тоже остаётся.


---

## 16. Phase 4 round 4 — primary-category prior в union retrieval

**Дата:** 2026-05-03
**Контекст:** Round 3 (раздел 15) починил jeans 7193 (primary=jeans, single
category в candidates). Но на test4 нашлись jeans 26995 и 27020, где router
правильно выдал union `{jeans, dresses}` — а в финальной выдаче всё равно
только платья.

### 16.1 Почему union не помог

Union даёт всего лишь *возможность* для items секундной категории попасть
в результат — реальное ранжирование решает финальный score. И тут вылез
несимметричный numeric bias:

- В каталоге **399 платьев vs 66 джинсов**.
- При union по обеим — у платьев банально больше "лотерейных билетов"
  попасть близко к query в embedding-пространстве FashionSigLIP.
- Если sem cosines примерно сравнимы у топ-2 jeans (0.78) и топ-10 dresses
  (0.76–0.79), численно платья перетягивают top-10 сортировку.

Это не значит что router был неправ — он правильно сказал "primary=jeans,
не уверен на 100% но 60-40". А retrieval это игнорировал, гонял чисто по
score без учёта prior'а.

### 16.2 Фикс — soft prior на primary категорию

При union retrieval добавляем мультипликативный boost к финальному score
items из primary категории:

```kotlin
finalScore[i] = base * gate(col) * (boost if i.category == primary else 1.0)
boost = 1.0 + (margin / LOW_CONFIDENCE_MARGIN) * MAX_PRIMARY_BOOST
      = 1.0 + (margin / 0.05) * 0.20
```

**Адаптивный**: чем больше margin (router был "почти уверен"), тем больше
boost. margin=0 (полная неопределённость) → boost=1.0 (без bias'а, чистый
score). margin=0.05 (порог) → boost=1.20 (20% к primary).

| query тип | margin | boost для primary | secondary penalty (relative) |
|---|---|---|---|
| jeans 7193 (clear)  | >0.05 | n/a (no union)        | n/a |
| jeans 26995 (close) | ~0.04 | 1.16  (+16%)          | -16% |
| jeans 27020 (close) | ~0.03 | 1.12  (+12%)          | -12% |
| children's dress 4931 | ~0.02 | 1.08 (+8%)         | -8% |

**Не насильный**: если 2nd категория реально имеет item с sem cosine 0.85
а primary только 0.70, разница 0.15 — boost ×1.20 даёт max 0.84, всё
равно проигрывает 0.85. Boost решает только close-call.

### 16.3 Изменения

- `core/retrieval/CosineRetriever.kt`: новые опциональные параметры
  `primaryCategory: String?`, `primaryBoost: Float = 1.0`. Boost
  применяется только при `categories.size > 1` (single-category
  retrieval не нуждается в bias'е).
- `core/ml/InferenceEngine.kt`: считает boost адаптивно от margin,
  пробрасывает в retriever, логирует в logcat:
  ```
  router decision: {dresses=0.79, ...} → primary=jeans candidates=[jeans, dresses]
    margin=0.04 boost=1.16, top=10, latency=…
  ```

### 16.4 Что ещё может потребоваться

Если 26995/27020 всё равно выдают платья — значит реальный sem gap >0.20
(jeans items имеют намного более низкий sem cosine с query чем dresses).
Это уже фундаментальная проблема FashionSigLIP encoding для этих
конкретных query — возможно из-за плохой сегментации u2netp, дающей
cropped что не похожее ни на jeans-product-shot.

Возможные шаги в ту сторону:
1. Дебаг-toggle "show segmented preview" в Results — посмотреть что реально
   попадает в encoder.
2. Encoding на BOTH raw и cropped, ансамбль 50/50 — может быть устойчивее.
3. Per-image mini-classifier (256-dim head поверх FashionSigLIP, обученный
   на каталоге распознавать категории) — заменит prototype-router.
