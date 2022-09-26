# What is that:
A quick and crude demo of how can be neural brush implemented using a U-net 


## How to use:

as always: 
```
pip install -r Requirements.txt
```

and run neural_brush.py
inside opencv windows the following keys are valid: 

Esc - quit  
1 - positive labeling brush  
2 - negative labeling brush  
3 - neural brush(available after train procedure)  
t - train now for one epoch  
s - save model  
l - load model  

The logic is the following - label with positive brush anything that should be segmented for sure in final output and label with negative brush enything that should be ommited from final output.  Train net for single time and switch to "3" to check the resulting neural brush.
DEMO:
https://youtu.be/iROaBjuMvN0
## known problems:

-global namespacing should be avoided  
-opencv as a GUI is a bad idea  
-opencv GUI might not work in macOS  
-recent tensorflow.keras like 2.8.0  do have a resizing and rescaling layers, no need to use lambda layer anymore