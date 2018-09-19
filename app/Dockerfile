# Python + Torch
# Thanks to the folks at Floydhub, this is a solved problem.
FROM floydhub/pytorch:0.3.1-py2.34

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

# Run the app.  CMD is required to run on Heroku
# $PORT is set by Heroku			
CMD gunicorn --bind 0.0.0.0:$PORT wsgi 

# RUN adduser -D myuser
# USER myuser

# Run the app.  CMD is required to run on Heroku
# $PORT is set by Heroku			
# CMD gunicorn --bind 0.0.0.0:$PORT app 