# Thesis_ConvLSTM

To incorporate the encoder-forecaster structure in ConvLSTM, one need delete line 271 and 272 in convolutional_recurrent.py and run the modified convolutional_recurrent.py before build the ConvLSTM encoder-forecaster network.

Traditional ROVER algorithm from weather nowcasting community can not handle the growing case in precipitation nowcasting:

| <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/Input_5_frames.gif" width="200" height="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/truth_15_frames.gif" width="200" height="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/rover_15_frames.gif" width="200" height="200" /> |  
|:--:|:--:|:--:| 
| *Input frames* | *Ground truth* | *ROVER prediction* |

We may think of this problem from the machine learning perspective:

Periodic observations taken from a dynamic system. Each radar echo map is a spatial $N_1\times N_2$ grid. Each pixel has an integer between 0 and 255 representing the rainfall intensity:
 <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/PrecipitationModel-1.jpg" width="700"  /> 








