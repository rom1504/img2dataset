FROM gitpod/workspace-full:latest

RUN sudo apt-get update && sudo apt-get install -y python3-opencv
