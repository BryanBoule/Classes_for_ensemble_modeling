# Indicate where we are getting the image (from dockerhub and nvidia's repo)
FROM python:3.7-slim

# Add a new user
RUN useradd -m user_ubuntu

# make our user own its own home directory
RUN chown -R user_ubuntu:user_ubuntu /home/user_ubuntu/

# copy all file from this directory to a directory called app inside the
# home of user and user owns it
COPY --chown=user_ubuntu *.* /home/user_ubuntu/app/

# change to user: user
USER user_ubuntu

# Install dependencies
RUN cd /home/user_ubuntu/app/ && pip3 install -r requirements.txt

WORKDIR /home/user_ubuntu/app

# Run script automatically when image is built
RUN python3 main.py