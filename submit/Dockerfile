FROM python:3.8-slim
WORKDIR /all
ADD ass1_server.py /all
ADD client /all
ADD object_detection /all
RUN apt-get update && apt-get install ffmpeg  -y
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["python", "/all/ass1_server.py"]
