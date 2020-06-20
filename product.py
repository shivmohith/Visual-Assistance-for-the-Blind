from model import load_vqa_model, load_image_caption_model, generate_caption, predict_answer
from conversions import listen_to_user, inform_user
import cv2

import en_core_web_md

'''
Loads the VideoCapture object
'''
cam = cv2.VideoCapture(0)

def capture_image():
    '''Captures the frame and saves it as a jpg image
    '''

    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame.jpg"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            cam.release()
            cv2.destroyAllWindows()
            return
    

print('loading word2vec')
nlp = en_core_web_md.load()

print('loading vqa model')
vqa_model = load_vqa_model()

print('loading image caption model')
encoder, decoder, image_features_extract_model, tokenizer, max_length = load_image_caption_model()

'''
Infinite loop giving the user the options to ask a question or hear about the surrounding
'''

while True:
    option = input('Enter c to describe the scene \n v to ask question')

    if option == 'v':
        capture_image()
        question = listen_to_user()
        print(question + '?')
        # cv2.imshow("image_captured",image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        answer = predict_answer(vqa_model, nlp, question + '?', "opencv_frame.jpg")
        print('\n \n ANSWER:',answer)
    else:
        capture_image()
        caption, _ = generate_caption("opencv_frame.jpg", encoder, decoder, image_features_extract_model, tokenizer, max_length)
        caption = ' '.join(caption[:-1])
        print('\n \n CAPTION:',caption)
        
        inform_user(caption)
