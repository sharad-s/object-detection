# MobileSSD-RaspberryPi-NCS-Script

Custom code for running a MobileNet SSD model on a Raspberry Pi using the Movidius NCS.

* `graph/graph` is the graph compiled by the NCSDK using the SSD model written in Caffe.
* `mobileSSD-pi.py` is custom code written to load the graph from filesystem, upload it to the NCS, and process predictions for realtime video input using the NCS.

## Requirements

* Raspberry Pi 3B+
* Connected USB2.0 Webcam.
* NCSDK and OpenCV installed on your Raspberry Pi.

## Instructions

* Make Sure USB 2.0 Webcam and Movidius Compute Stick are plugged into your Raspberry Pi.
* Run the `mobileSSD-pi.py` file from your Pi.
