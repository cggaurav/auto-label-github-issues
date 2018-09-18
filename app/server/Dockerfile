# Python + Torch
# Thanks to the folks at Floydhub, this is a solved problem.
FROM floydhub/pytorch:0.3.1-py2.34

# Flask
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]