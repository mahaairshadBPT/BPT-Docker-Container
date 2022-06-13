FROM osgeo/gdal:ubuntu-small-3.3.3

# We copy just the requirements.txt first to leverage Docker cache
WORKDIR /app
RUN apt-get update && apt-get install python3-pip -y && apt-get install python3-opencv -y

COPY ./model/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY ./model/src /app
