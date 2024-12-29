FROM ultralytics/ultralytics:latest-arm64

WORKDIR /app

RUN pip install --upgrade pip
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./model ./model
COPY ./bag_counter_pipeline.py ./bag_counter_pipeline.py
COPY ./entrypoint.py ./entrypoint.py

ENTRYPOINT ["python", "entrypoint.py"]