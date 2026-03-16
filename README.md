# 🤖 Бот для классификации пород керна

![QR Code](rock_bot_qr.png)

Сканируй QR-код → откроется [@RockClassificationBot](https://t.me/RockClassificationBot)

---

## 📁 1. Датасет

Разместите фотографии в папках согласно структуре:

```text
dataset/
├── train/           # Для обучения
│   ├── limestone/   # Известняк
│   ├── sandstone/   # Песчаник
│   └── shale/       # Сланец
├── val/             # Для валидации
└── test/            # Для теста
```

## 🧠 2. Обучение
```
pip install -r requirements.txt
python train.py
```

Модель сохранится в папку model/rock_classifier.pth.

## 🔑 3. Токен бота
Напишите @BotFather в Telegram.
Выполните команду /newbot.
Скопируйте токен (например: 123456789:ABC...).

## 🚀 4. Запуск
```bash
export BOT_TOKEN="ваш_токен_здесь"
python bot.py
```

Готово! Отправляйте боту фото керна — он определит породу.