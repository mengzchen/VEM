#!/usr/bin/env bash

conda create -n digirl_online python==3.10 -y

sudo apt-get update
cd ~ && mkdir install-android/ && cd install-android
wget https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u412-b08/openlogic-openjdk-8u412-b08-linux-x64-deb.deb
sudo apt install ./openlogic-openjdk-8u412-b08-linux-x64-deb.deb

sudo update-alternatives --config java

wget https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip

export ANDROID_HOME=/home/aiscuser/.android
mkdir -p $ANDROID_HOME
sudo apt-get install unzip
unzip sdk-tools-linux-4333796.zip -d $ANDROID_HOME