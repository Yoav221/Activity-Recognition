# Activity-Recognition
The goal in this project is to identify specific activity in a video, count how many times it accurs, and extract the location of it.

class1 = activity

class2 = non_activity

We use CNN + RNN archcitecture to extract features from the video's frames, and feed those features into the RNN.
Here we try to recognize an activity of size 10x10 that happens in a 10 frames window.
You can change it to your need in the const.py module and adjust the Network's parameters.

In order to detect the activity in a given video, we need to split the whole video to small samples, and the model will make a prediction on each sample. This way we get the location of all the samples and count how many samples the model recognized as class1 (the wanted activity). 


Instructions:

Set the pathes this way:

        Data -> Video_data -> class_1 -> 'class_1_1.avi'
                                         'class_1_2.avi'
        
                              class_2 - > 'class_2_1.avi'
                                          'class_2_2.avi'
        
                Train_data -> 'The code will create this data'                       
                Sample_data -> 'The code will create this data'                       
                      
                                 
- Adjust the pathes and parameters in 'const.py'
- Run 'data_preprocessing.py' to make the Data ready for training
- Run 'train.py' to train the model.
- Choose an Input Video, and run 'create_samples.py' to create the samples from the video.
- Run 'predict' to predict each sample.
- Run plot_prediction.py to see 3D map for all the activities (class1) our model found in the video.
- Run draw_rectangle.py to draw a rectangle around every activity in the video.
                                
  
  
- In 'models.py' we create the cnn+rnn architecture and create our custom Dataset class.
- In 'functions.py' we have useful functions that we'll use throughout the code.
