# Activity-Recognition
The goal in this project is to identify specific activity in a video, and count how many times it accurs, and also extract the location of each activity the model recognized.

In this specific project I train the model to recognize between explosion and non-explosion.
You can use the code also to your needs - you just need to change to change the settings in the 'const' module.

Just set the pathes like this:
Data -> Video_data -> class_1 -> 'class_1_1.avi'
                                 'class_1_2.avi'
        
                      class_2 - > 'class_1_1.avi'
                                  'class_1_2.avi'
        
        Train_data -> 'The code will create this data'                       
        Sample_data -> 'The code will create this data'                       
                      
                                 

- Run 'data_preprocessing.py' to make the Data ready for training
- Run 'train.py' to train the model.
- Choose an Input Video, and run 'predict.py'.
- Run plot_prediction.py to see the map for all the activities our model found in the video.
- Run draw_rectangle.py to draw a rectangle around every activity in the video.
                                
                                
- In 'models.py' we create the cnn+rnn architecture and create our custom Dataset class.
- In 'functions.py' we have useful functions that we'll use through the code.
