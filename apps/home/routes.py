from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required, current_user
from jinja2 import TemplateNotFound

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
@login_required
def beranda():
    username = get_username()
    return render_template('home/index.html', segment='index', username=username)


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