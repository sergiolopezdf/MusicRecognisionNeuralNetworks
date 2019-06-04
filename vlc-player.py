import vlc

player = vlc.MediaPlayer('1_15_SOS.mp3')

while (True):
    player.play()
    # print(player.get_time())
