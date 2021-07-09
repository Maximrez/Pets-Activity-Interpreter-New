import telebot
from functions import *

project_dir = r'D:\PycharmProjects\CV_project'
data_dir = os.path.join(project_dir, 'data')
yolo_dir = os.path.join(data_dir, 'yolov3')
telegram_dir = os.path.join(data_dir, 'telegram')

if not os.path.exists(telegram_dir):
    os.mkdir(telegram_dir)

TOKEN = "1899424187:AAFZ9OAVDhJWGIfB3eAS1hV3XJ7Coi7l3KY"
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def command_start(m):
    cid = m.chat.id
    bot.send_message(cid, "Привет! Отправь мне видео, а я его обработаю!")


@bot.message_handler(content_types=['video'])
def get_video(m):
    cid = m.chat.id
    file_info = bot.get_file(m.video.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = os.path.join(telegram_dir, m.video.file_id + ".mp4")
    with open(src, 'wb') as f:
        f.write(downloaded_file)
    bot.send_message(cid, "Видео обрабатывается...")

    out_path = os.path.join(telegram_dir, str(m.video.file_id) + ".avi")
    stats_m, stats_a = process_video(src, project_dir, out_path)

    # new_out_path = os.path.join(telegram_dir, "converted_" + str(m.video.file_id) + ".mp4")
    # ff_bin_path = r'C:\Users\Максим\ffmpeg-2021-07-06-git-758e2da289-full_build\bin'

    # convert_avi_to_mp4(out_path, new_out_path, ff_bin_path, width, height, fps)

    with open(out_path, 'rb') as f:
        bot.send_video(cid, f, supports_streaming=True)

    graph_path = os.path.join(telegram_dir, "graph_" + str(m.video.file_id) + ".png")
    stats_graph(stats_m, stats_a, graph_path)

    with open(graph_path, 'rb') as f:
        bot.send_photo(cid, f, "Статистика активности животного")

    stats_config_path = os.path.join(project_dir, 'presentation', 'colors_config.png')
    with open(stats_config_path, 'rb') as f:
        bot.send_photo(cid, f, "Описание статистики")


bot.polling(none_stop=True, interval=0)
