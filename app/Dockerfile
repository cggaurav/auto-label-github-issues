# Python + Torch
# Thanks to the folks at Floydhub, this is a solved problem.
# FROM floydhub/pytorch:0.3.1-py2.34
# Though the above is just so heavy

FROM python:2.7.15-jessie

RUN python -v
RUN pip --version
# Flask
# Install python and pip
ADD ./server/requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -q -r /tmp/requirements.txt

# Add our code
ADD ./server /opt/server/
WORKDIR /opt/server

# Expose is NOT supported by Heroku
# EXPOSE 5000

# Run the image as a non-root user
RUN adduser --disabled-l myuser
USER myuser

# Run the app.  CMD is required to run on Heroku
# $PORT is set by Heroku			
CMD gunicorn --bind 0.0.0.0:$PORT wsgi 