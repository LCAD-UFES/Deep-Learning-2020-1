# Sleeping Detection Project for Deep Learning Class
#### Ufes
###### Cezar Augusto Gobbo Passamani
#

#### Ubuntu 18.04 LTS (64 bit) Instalation Guide

##### Dependencies

sudo apt-get install build-essential \
git  \
wget  \
swig  \
libgtk2.0-dev  \
libgtkmm-2.4-dev  \
libcurl4-openssl-dev  \
pkg-config  \
lighttpd \
libsqlite3-dev  \
sqlite3 \
unzip \
python-dev  \
python-numpy  \
libtbb2  \
libtbb-dev  \
libjpeg-dev  \
libpng-dev \
libtiff-dev \
libdc1394-22-dev \
libncurses5-dev \
libprotobuf-dev  \
libleveldb-dev \
libsnappy-dev \
libhdf5-serial-dev \
protobuf-compiler \
--no-install-recommends libboost-all-dev \
libatlas-base-dev \
libgflags-dev  \
libgoogle-glog-dev \
liblmdb-dev \
libblas-dev  \
libopenblas-base \
liblapack-dev \
libvpx-dev \
yasm  \
libx264-dev \
libbluetooth-dev \
libssl-dev \
libstxxl-dev  \
libcurl3 \
php7.2-sqlite3  \
php7.2-cgi  \
php7.2-zip  \
php7.2-curl
```

Cmake
```
mkdir ~/libraries
cd ~/libraries
git clone https://gitlab.kitware.com/cmake/cmake.git
cd cmake
./bootstrap
make
sudo make install
sudo ldconfig
cd ~/libraries
```

OPENBLAS Installation
```
mkdir ~/libraries
cd ~/libraries
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
git checkout optimized_for_deeplearning
nano Makefile.rule

Uncomment line: # TARGET = PENRYN
Change to: TARGET = HASWELL

make
sudo make install
sudo ldconfig
cd ~/libraries
```

OpenCV 3.4.0 Installation
```sh
mkdir ~/libraries
cd ~/libraries
mkdir opencv3.4.0 && cd opencv3.4.0
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.0
cd ../opencv
git checkout 3.4.0
mkdir build && cd build
cmake -DWITH_EIGEN=ON -DWITH_V4L=ON -DWITH_TBB=ON -DWITH_OPENGL=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_opencv_cudacodec=OFF -DENABLE_PRECOMPILED_HEADERS=ON -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=~/libraries/opencv3.4.0/opencv_contrib/modules  ~/libraries/opencv3.4.0/opencv/
make -j12
sudo make install
sudo ldconfig
cd ~/libraries
```
CAFFE Installation
```sh
mkdir ~/libraries
cd ~/libraries
git clone https://github.com/BVLC/caffe.git
cd ~/libraries/caffe/
protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir ~/libraries/caffe/include/caffe/proto
mv ~/libraries/caffe/src/caffe/proto/caffe.pb.h ~/libraries/caffe/include/caffe/proto
mkdir build && cd build
cmake -DUSE_OPENCV=OFF -DBLAS=open -DCPU_ONLY=ON -DBUILD_docs=OFF -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE ..
make -j12
sudo make install
sudo ldconfig
cd ~/libraries
```

Dlib Installation
```sh  
mkdir ~/libraries
cd ~/libraries
wget http://dlib.net/files/dlib-19.10.tar.bz2
tar -xvjf dlib-19.10.tar.bz2
cd ~/libraries/dlib-19.10/
mkdir build
cd build
cmake -DDLIB_USE_CUDA=OFF ..
make -j12
sudo make install
sudo ldconfig
cd ../..
rm dlib-19.10.tar.bz2
cd ~/libraries
```
#
##### Shape Predictor

> Download the Shape Predictor from the link: https://drive.google.com/file/d/1wOSbdgNue8bjag3MC8_jHBiG0qo9r7rm

> Place it in the directory SleepingDetection/net_models/faces/ 

#
##### Building Project

```sh
cd sleeping detection
mkdir build && cd build
cmake ..
make
```
#
##### Usage
```sh
Usage: ./sleeping_detection [--camera CAMERA] [--file FILE] [--path PATH] [--width WIDTH] [--height HEIGHT]
```
`[--camera CAMERA]` means any camera attached to the PC, just pass as an argument using `-c`. Example:
```
./sleeping_detection -c 0
```
`[--file FILE]` means if you have any video file to run, just pass as an argument using `-f`. Example
```
./sleeping_detection -f video.mp4
```
`[--path PATH]` means if you have a folder containing many videos, just pass the folder path as an argument using `-p`. Example:
```
./sleeping_detection -p /path/to/videos/
```
`[--width WIDTH] [--height HEIGHT]` means the width and the height of the cam or video you are passing as an argument. To set it, just pass the values as arguments using `-w` and `-h`. Example:
```
./sleeping_detection -p /path/to/videos/ -w 1280 -h 720
```