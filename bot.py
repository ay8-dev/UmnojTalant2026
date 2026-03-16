import os
import io
import json
import logging
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
TOKEN = os.getenv("BOT_TOKEN", "8702367143:AAEaKMsA13xzzNpNCUOzGosd-xJ_eyjxvQM")
MODEL_PATH = "model/rock_classifier.pth"
METADATA_PATH = "model/metadata.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES_RU = {
    "limestone": "🪨 Известняк",
    "sandstone": "🟫 Песчаник", 
    "shale": "⬛ Сланец (аргиллит)"
}

ROCK_DESCRIPTIONS = {
    "limestone": "Осадочная порода органического происхождения, состоит преимущественно из кальцита (CaCO₃). Образуется в морских и пресноводных бассейнах.",
    "sandstone": "Обломочная осадочная порода, состоящая из зёрен кварца и других минералов, скреплённых цементом. Характерна для древних речных и дельтовых отложений.",
    "shale": "Тонкозернистая осадочная порода, состоящая из глинистых минералов. Образуется в условиях спокойного осадконакопления (озёра, морские мелководья)."
}


class RockClassifier:
    def __init__(self):
        self.model = None
        self.classes = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.load_model()
    
    def load_model(self):
        """Загрузка обученной модели"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. "
                                   f"Сначала запустите train.py")
        
        # Загружаем метаданные
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                self.classes = metadata.get('classes', ['limestone', 'sandstone', 'shale'])
        else:
            self.classes = ['limestone', 'sandstone', 'shale']
        
        # Создаем и загружаем модель
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.model.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, len(self.classes))
        )
        
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        logger.info(f"Модель загружена. Классы: {self.classes}")
    
    def predict(self, image: Image.Image) -> dict:
        """Классификация изображения"""
        # Преобразуем изображение
        image_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Получаем топ-3 предсказания
            probs, indices = torch.topk(probabilities, k=min(3, len(self.classes)))
            
            results = []
            for prob, idx in zip(probs[0], indices[0]):
                class_name = self.classes[idx.item()]
                results.append({
                    'class': class_name,
                    'class_ru': CLASS_NAMES_RU.get(class_name, class_name),
                    'probability': prob.item() * 100
                })
            
            return {
                'top_prediction': results[0],
                'all_predictions': results
            }


# Инициализация классификатора
classifier = RockClassifier()


# ===== Обработчики команд Telegram =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды /start"""
    welcome_text = (
        "👋 *Привет! Я бот для классификации пород керна.*\n\n"
        "Я могу определить тип породы по фотографии керна:\n"
        "• 🪨 Известняк (*limestone*)\n"
        "• 🟫 Песчаник (*sandstone*)\n" 
        "• ⬛ Сланец/аргиллит (*shale*)\n\n"
        "📸 Просто отправь мне фотографию керна, и я определю его тип!\n\n"
        "Используй /help для дополнительной информации."
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды /help"""
    help_text = (
        "📋 *Справка по использованию:*\n\n"
        "*Как пользоваться:*\n"
        "1. Сделайте фото керна при хорошем освещении\n"
        "2. Отправьте фото боту\n"
        "3. Получите результат классификации\n\n"
        "*Доступные команды:*\n"
        "/start — Начало работы\n"
        "/help — Эта справка\n"
        "/info — Информация о модели\n\n"
        "*Советы для лучшего результата:*\n"
        "• Фотографируйте при дневном свете\n"
        "• Держите камеру перпендикулярно поверхности\n"
        "• Избегайте бликов и теней\n"
        "• Крупный план текстуры породы улучшает точность"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Информация о модели"""
    info_text = (
        "🧠 *Информация о модели:*\n\n"
        f"• Архитектура: *ResNet-18*\n"
        f"• Число классов: *{len(classifier.classes)}*\n"
        f"• Устройство: *{DEVICE}*\n\n"
        f"*Поддерживаемые классы:*\n"
    )
    for cls in classifier.classes:
        info_text += f"• {CLASS_NAMES_RU.get(cls, cls)}\n"
    
    await update.message.reply_text(info_text, parse_mode='Markdown')


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка входящих фотографий"""
    try:
        # Отправляем сообщение о начале обработки
        processing_msg = await update.message.reply_text(
            "🔍 Анализирую изображение..."
        )
        
        # Получаем фото с наилучшим качеством
        photo = update.message.photo[-1]
        photo_file = await context.bot.get_file(photo.file_id)
        
        # Загружаем изображение в память
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(io.BytesIO(photo_bytes)).convert('RGB')
        
        # Классификация
        result = classifier.predict(image)
        top = result['top_prediction']
        
        # Формируем ответ
        confidence_emoji = "🟢" if top['probability'] > 80 else "🟡" if top['probability'] > 50 else "🔴"
        
        response = (
            f"*Результат классификации:*\n\n"
            f"{confidence_emoji} *{top['class_ru']}*\n"
            f"   Уверенность: *{top['probability']:.1f}%*\n\n"
            f"📖 *Описание:*\n{ROCK_DESCRIPTIONS.get(top['class'], '')}\n\n"
            f"*Другие варианты:*\n"
        )
        
        # Добавляем остальные предсказания
        for pred in result['all_predictions'][1:]:
            bar = "█" * int(pred['probability'] / 10) + "░" * (10 - int(pred['probability'] / 10))
            response += f"• {pred['class_ru']}: {bar} {pred['probability']:.1f}%\n"
        
        # Удаляем сообщение о обработке и отправляем результат
        await processing_msg.delete()
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text(
            "❌ *Ошибка при обработке изображения.*\n"
            "Попробуйте отправить другое фото или повторите позже.",
            parse_mode='Markdown'
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    await update.message.reply_text(
        "📸 Пожалуйста, отправьте *фотографию керна* для классификации.\n"
        "Используйте /help для справки.",
        parse_mode='Markdown'
    )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка ошибок"""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ Произошла ошибка. Попробуйте позже."
        )


def main():
    """Запуск бота"""
    if TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Ошибка: Установите переменную окружения BOT_TOKEN")
        print("   Пример: export BOT_TOKEN='your_token_here'")
        return
    
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("info", info_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)
    
    logger.info("Бот запущен!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()