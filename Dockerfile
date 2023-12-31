FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY . /app

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "chat.py", "--server.port=8501", "--server.address=0.0.0.0"]