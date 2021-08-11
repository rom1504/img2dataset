FROM gitpod/workspace-full:latest

RUN apt-get update && apt-get install -y python3-opencv
