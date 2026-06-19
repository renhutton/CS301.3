CS301.3 MVP Development
The MVP development of an artificial intelligence-based gesture-controlled system designed to control computer interfaces without physical touch. 
The intended use of this software is to aid surgeons and medical professionals in sterile environments

***Requirements for Deployment***
This project runs on dedicated hardware, and will not work by just running the code
to set up, run "withUItracking.py" on a raspberry pi 5 which is connected to a 7 inch touch screen and Pi Camera and have a serial adaptor connected to the USB port on you windows PC.
On the windows PC, run "reciever_app.py" to start the listener. 

when both parts of the MVP are running, the Pi will send serial commands to the PC, which the listener program will pick up and turn into mouse and keyboard commands.

the right hand pointer finger controls the mouse, and the left hand controls inputs such as:
Single pinch for click
double pinch for double click
hold pinch for drag click
index pinch for right click
both pinch and move hand farther or closer to the screen for scroll.
