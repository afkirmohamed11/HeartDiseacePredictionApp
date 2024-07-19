FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "python" ]

CMD ["app.py"]