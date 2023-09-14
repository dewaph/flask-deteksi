from apps.home import blueprint
from flask import render_template, request, current_app, Response
from flask_login import login_required, current_user
from jinja2 import TemplateNotFound
import os
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips

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
@login_required
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