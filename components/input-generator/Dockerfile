FROM python:3.6

COPY requirements.txt requirements.txt
COPY src/create_input.py src/create_input.py

# Install Python Requirements
RUN pip3 install -r requirements.txt

EXPOSE 8888

# Start program
ENTRYPOINT ["python", "src/create_input.py"]