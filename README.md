# Thesis_ConvLSTM

To incorporate the encoder-forecaster structure in ConvLSTM, one need delete line 271 and 272 in convolutional_recurrent.py and run the modified convolutional_recurrent.py before build the ConvLSTM encoder-forecaster network.

## Traditional ROVER algorithm from weather nowcasting community can not handle the growing case in precipitation nowcasting:

| <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/Input_5_frames.gif" width="200" height="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/truth_15_frames.gif" width="200" height="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/rover_15_frames.gif" width="200" height="200" /> |  
|:--:|:--:|:--:| 
| *Input frames* | *Ground truth* | *ROVER prediction* |



