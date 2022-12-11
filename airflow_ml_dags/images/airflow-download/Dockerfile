FROM python:3.9
RUN python3 -m pip install --upgrade pip
COPY download.py requirements.txt /
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "download.py"]
