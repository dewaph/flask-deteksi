from apps.home import blueprint
from flask import render_template, request, current_app, Response, jsonify, send_file
from flask_login import login_required, current_user
from jinja2 import TemplateNotFound
import os
import io
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from gtts import gTTS

def resize_image(image, size):
    resized_image = image.resize(size)
    return resized_image

def get_username():
    return current_user.username if current_user.is_authenticated else None

def get_context(segment):
    context = {'segment': segment}
    context['username'] = current_user.username if current_user.is_authenticated else None
    return context

@blueprint.route('/beranda')
def beranda():
    username = get_username()
    return render_template('home/index.html', segment='index', username=username)


@blueprint.route('/video_chunk')
def video_chunk():
    combined_video_path = os.path.join(current_app.root_path, 'static', 'kamusku', 'combined_video.mp4')

    start_byte = int(request.args.get('start', 0))
    chunk_size = 1024 * 1024  # Ubah sesuai kebutuhan
    
    with open(combined_video_path, 'rb') as video_file:
        video_file.seek(start_byte)
        chunk = video_file.read(chunk_size)
        
    return Response(chunk, content_type='video/mp4')

@blueprint.route('/kamus', methods=['GET', 'POST'])
def kamus():
    output_folder = os.path.join(current_app.root_path, 'static', 'kamusku')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if request.method == "POST":
        user_input = request.form.get("user_input").upper()

        num_columns = 4  # Jumlah kolom yang diinginkan
        
        video_clips = []

        if user_input:
            alphabet_chunks = [user_input[i:i + num_columns] for i in range(0, len(user_input), num_columns)]
            image_number = 0
            for chunk in alphabet_chunks:
                for i, letter in enumerate(chunk):
                    image_number += 1
                    image_path = os.path.join(output_folder, f"{letter}.png")

                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        resized_image = resize_image(image, (200, 200))
                        resized_image.save(image_path)

                    video_path = os.path.join(output_folder, f"{letter}.mp4")

                    if os.path.exists(video_path):
                        video_clip = VideoFileClip(video_path)
                        video_clips.append(video_clip)

            if video_clips:
                combined_clip = concatenate_videoclips(video_clips)
                combined_video_file = os.path.join(output_folder, "combined_video.mp4")
                combined_clip.write_videofile(combined_video_file, codec="libx264", audio_codec="aac")

                combined_video_path = os.path.join(output_folder, "combined_video.mp4")
                return render_template("home/kamus.html", user_input=user_input, combined_video_path=combined_video_path)
            else:
                return render_template("home/kamus.html", user_input=user_input, error_message="Data tidak ditemukan untuk kata ini.")

    else:
        alphabet_set = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        letters_data = []

        for letter in sorted(alphabet_set):
            letter_data = {"letter": letter, "images": [], "videos": []}

            image_path = os.path.join(output_folder, f"{letter}.png")

            if os.path.exists(image_path):
                letter_data["images"].append(f"{letter}.png")

            video_path = os.path.join(output_folder, f"{letter}.mp4")

            if os.path.exists(video_path):
                letter_data["videos"].append(f"{letter}.mp4")

            letters_data.append(letter_data)

        return render_template("home/kamus.html", letters_data=letters_data)
    

@blueprint.route('/terjemahan', methods=['GET'])
def terjemahan():
    try:
        with open("transcript.txt", "r") as f:
            transcript_content = f.read()
    except FileNotFoundError:
        transcript_content = "Belum ada Terjemahan"
    with open("transcript.txt", "w") as f:
        f.truncate(0)
    return render_template('home/terjemahan.html', transcript_content=transcript_content)

@blueprint.route('/add_space', methods=['POST'])
def add_space():
    global start_time
    try:
        with open("transcript.txt", "a") as f:
            f.write(" ")  
        start_time = None
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Spasi berhasil ditambahkan ke file."}), 200

# def generate_frames():
#     global cap, start_time
#     # Inisialisasi webcam dan variabel lainnya
#     cap = cv2.VideoCapture(0)
#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils
#     mp_drawing_styles = mp.solutions.drawing_styles

#     hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

#     model_dict1 = pickle.load(open('apps/static/model/model1.p', 'rb'))
#     model_dict2 = pickle.load(open('apps/static/model/model2.p', 'rb'))
#     model1 = model_dict1['model1']
#     model2 = model_dict2['model2']

#     labels_dict1 = {0: 'C', 1: 'E', 2: 'I', 3: 'J', 4: 'L', 5: 'O', 6: 'R', 7: 'U', 8: 'V', 9: 'Z'}
#     labels_dict2 = {0: 'A', 1: 'B', 2: 'D', 3: 'F', 4: 'G', 5: 'H', 6: 'K', 7: 'M', 8: 'N', 9: 'P',
#                         10: 'Q', 11: 'S', 12: 'T', 13: 'W', 14: 'X', 15: 'Y'}

#     start_time = None

#     while True:
#         data_aux = []
#         x_ = []  # Initialize the x_ list for hand landmarks
#         y_ = []  # Initialize the y_ list for hand landmarks

#         ret, frame = cap.read()

#         if not ret:
#             break

#         cv2.putText(frame, 'Mulai Menerjemahkan', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
#                     cv2.LINE_AA)

#         H, W, _ = frame.shape

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         results = hands.process(frame_rgb)
#         if results.multi_hand_landmarks:
#             n = len(results.multi_hand_landmarks)
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame,  # image to draw
#                     hand_landmarks,  # model output
#                     mp_hands.HAND_CONNECTIONS,  # hand connections
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())

#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y

#                     x_.append(x)
#                     y_.append(y)

#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))

#             if n == 1:
#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10

#                 x2 = int(max(x_) * W) - 10
#                 y2 = int(max(y_) * H) - 10

#                 if start_time is None:
#                     start_time = time.time()  # Set waktu awal jika belum diinisialisasi

#                 if start_time is not None and time.time() - start_time >= 1.5:
#                     prediction1 = model1.predict([np.asarray(data_aux)])
#                     predicted_character = labels_dict1[int(prediction1[0])]


#                     # Tambahkan hasil prediksi huruf selanjutnya
#                     if predicted_character != ' ':
#                         with open("transcript.txt", "a") as f:
#                             f.write(predicted_character)

#                     start_time = None  # Reset waktu awal

#             else:
#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10

#                 x2 = int(max(x_) * W) - 10
#                 y2 = int(max(y_) * H) - 10

#                 if start_time is None:
#                     start_time = time.time()  # Set waktu awal jika belum diinisialisasi

#                 if start_time is not None and time.time() - start_time >= 1.5:
#                     prediction2 = model2.predict([np.asarray(data_aux)])
#                     predicted_character = labels_dict2[int(prediction2[0])]


#                     # Tambahkan hasil prediksi huruf selanjutnya
#                     if predicted_character != ' ':
#                         with open("transcript.txt", "a") as f:
#                             f.write(predicted_character)

#                     start_time = None  # Reset waktu awal
        
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
#     # Setelah loop selesai, lepaskan kamera
#     cap.release()

#     # Tutup semua jendela OpenCV
#     cv2.destroyAllWindows()

# @blueprint.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @blueprint.route('/close')
# def close():
#     # Setelah loop selesai, lepaskan kamera
#     cap.release()
#     with open("transcript.txt", "w") as f:
#         f.truncate(0)

#     # Tutup semua jendela OpenCV
#     cv2.destroyAllWindows()
#     return"tutup windows"
    
        
def generate_transcript():
    while True:
        with open("transcript.txt", "r") as f:
            transcript_content = f.read()
        yield f"data: {transcript_content}\n\n"

@blueprint.route('/transcript_updates')
def transcript_updates():
    return Response(generate_transcript(), mimetype='text/event-stream')

@blueprint.route('/convert_to_audio', methods=['POST'])
def convert_to_audio():
    data = request.json
    text = data.get('text')

    if text:
        tts = gTTS(text=text, lang='id', slow=False)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        
        return send_file(audio_io, mimetype='audio/mpeg', as_attachment=True, attachment_filename='converted_audio.mp3')
    else:
        return jsonify({"error": "Text is required"}), 400


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None