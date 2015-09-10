#!/bin/bash

sudo rm /usr/local/bin/mycaffe
sudo rm /usr/local/lib/libcaffe.a
sudo rm /usr/local/lib/libcaffe.so
sudo ln -s /home/withdove/Dropbox/Toolbox/caffe/.build_release/tools/caffe.bin /usr/local/bin/mycaffe
sudo ln -s /home/withdove/Dropbox/Toolbox/caffe/.build_release/lib/libcaffe.a /usr/local/lib/libcaffe.a
sudo ln -s /home/withdove/Dropbox/Toolbox/caffe/.build_release/lib/libcaffe.so /usr/local/lib/libcaffe.so
