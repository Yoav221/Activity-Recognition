# Activity-Recognition
The goal in this project is to identify specific activity in a video, count how many times it accurs, and extract the location of each activity the model recognized. My problem was to find explosions in a given video, but you can use this code to every problem you choose.
Just make sure to change the settings and parameters in the 'const.py' module.
We use CNN + RNN archcitecture to extract features from the video's frames, and feed those features into the RNN.

In order to get all the activities in a given video, we need to run our model on the whole video. This means we need to split the whole video to small samples, and the model will make a prediction on each sample. This way we get the location of all the samples and count how many samples the model recognized as explosions. 


Instructions:

Set the pathes this way:

        Data -> Video_data -> class_1 -> 'class_1_1.avi'
                                         'class_1_2.avi'
        
                              class_2 - > 'class_2_1.avi'
                                          'class_2_2.avi'
        
                Train_data -> 'The code will create this data'                       
                Sample_data -> 'The code will create this data'                       
                      
                                 

- Run 'data_preprocessing.py' to make the Data ready for training
- Run 'train.py' to train the model.
- Choose an Input Video, and run 'predict.py'.
- Run plot_prediction.py to see the map for all the activities our model found in the video.
- Run draw_rectangle.py to draw a rectangle around every activity in the video.
                                
                                
- In 'models.py' we create the cnn+rnn architecture and create our custom Dataset class.
- In 'functions.py' we have useful functions that we'll use through the code.
