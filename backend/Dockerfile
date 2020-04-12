FROM ubuntu:latest

RUN apt update -y
RUN apt install -y python3-pip
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt 
EXPOSE 5000
CMD ["python3","app.py"]