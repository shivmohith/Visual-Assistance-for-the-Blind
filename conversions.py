import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

'''
Loading a speech recognition object ans sets the energy threshold
'''

r = sr.Recognizer()
r.energy_threshold = 1000 # how sensitive the recognizer is to when recognition should start
#Higher values less sensitive - better in loud rooms

language = 'en-IN'

def listen_to_user():
    '''Returns a string
    Listens to user's voice and converts to text
    '''

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5) # listen for 0.5 second to calibrate the energy threshold for ambient noise levels
        print("SAY SOMETHING")
        audio = r.listen(source, timeout=3,phrase_time_limit=30)
        print("TIME OVER, THANKS")
    try:
        text = r.recognize_google(audio, language = 'en-IN')
        return text
    except:
        pass

def inform_user(sentence):
    ''' Converts text to audio and plays the audio

    Args:
        sentence: sentence in string format
    '''

    audio = gTTS(text=sentence, lang=language, slow=False) 
    audio.save("audio.mp3") 
    playsound('audio.mp3')