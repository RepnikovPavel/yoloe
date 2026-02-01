#!/bin/bash

CKPTDIR=${1:-"./models"}
MODEL_ID="jameslahm/yoloe"
# Создаем путь в стиле Hugging Face Cache
SNAPSHOT_DIR="$CKPTDIR/models--jameslahm--yoloe/snapshots/main"

echo "📁 Создаем структуру HF cache в $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"
cd "$SNAPSHOT_DIR"

echo "📥 Список файлов для загрузки..."
# Составляем массив файлов на основе предоставленного списка
FILES=(
    ".gitattributes"
    "README.md"
    "yoloe-11l-seg-coco-pe.pt"
    "yoloe-11l-seg-coco.pt"
    "yoloe-11l-seg-pf.pt"
    "yoloe-11l-seg.pt"
    "yoloe-11m-seg-coco-pe.pt"
    "yoloe-11m-seg-coco.pt"
    "yoloe-11m-seg-pf.pt"
    "yoloe-11m-seg.pt"
    "yoloe-11s-seg-coco-pe.pt"
    "yoloe-11s-seg-coco.pt"
    "yoloe-11s-seg-pf.pt"
    "yoloe-11s-seg.pt"
    "yoloe-v8l-seg-coco-pe.pt"
    "yoloe-v8l-seg-coco.pt"
    "yoloe-v8l-seg-pf.pt"
    "yoloe-v8l-seg.pt"
    "yoloe-v8m-seg-coco-pe.pt"
    "yoloe-v8m-seg-coco.pt"
    "yoloe-v8m-seg-pf.pt"
    "yoloe-v8m-seg.pt"
    "yoloe-v8s-seg-coco-pe.pt"
    "yoloe-v8s-seg-coco.pt"
    "yoloe-v8s-seg-pf.pt"
    "yoloe-v8s-seg.pt"
)

# Запускаем скачивание
for file in "${FILES[@]}"; do
    if [[ -n "$file" ]]; then
        echo "⏳ Загружаем $file"
        wget -q --show-progress --continue "https://huggingface.co/$MODEL_ID/resolve/main/$file"
    fi
done
wget -O $SNAPSHOT_DIR/mobileclip_blt.pt https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt

echo -e "\n✅ Проверка загруженных файлов:"
echo "Всего .pt файлов: $(ls *.pt 2>/dev/null | wc -l)"
du -sh *.pt 2>/dev/null | sort -h | head -5
echo "Общий размер:"
du -sh .

echo -e "\n🎉 Готово! Модель сохранена в:"
echo "path = \"$SNAPSHOT_DIR\""