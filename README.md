# Thesis_ConvLSTM

To incorporate the encoder-forecaster structure in ConvLSTM, one need delete line 271 and 272 in convolutional_recurrent.py and run the modified convolutional_recurrent.py before build the ConvLSTM encoder-forecaster network.

#### Traditional ROVER algorithm from weather nowcasting community can not handle the growing case in precipitation nowcasting (predcit future radar echo maps):

| <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/Input_5_frames.gif" width="200" height="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/truth_15_frames.gif" width="200" height="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/rover_15_frames.gif" width="200" height="200" /> |  
|:--:|:--:|:--:| 
| *Input frames* | *Ground truth* | *ROVER prediction* |

#### We may think of this problem from the machine learning perspective:

Periodic observations taken from a dynamic system. Each radar echo map is a spatial $ N_1 \times N_2 $ grid. Each pixel has an integer between 0 and 255 representing the rainfall intensity:
 <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/PrecipitationModel-1.jpg" width="500"  />
 
Inspired by LSTM neural networks which is good at capturing temporal relationships, we add convolution operation to LSTM at each time step to build so-called ConvLSTM. For ConvLSTM, we can keep input in 3D dimension so that we do not lose the spatial relationships:

| <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/conv_inner-1.jpg" width="400" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/ConvLSTM_Encoder_Forecaster-1.jpg" width="400"  /> |
|:--:|:--:|
| *Inner structure of ConvLSTM* | *Stacked ConvLSTM encoder-forecaster network* |

#### Prediction results from LSTM and ConvLSTM:

##### Growing case of radar echo maps: 
| <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/input_6427.gif" width="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/ground_truth_6427.gif" width="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/2layer_lstm_6427.gif" width="200" /> |  
|:--:|:--:|:--:|
| *Input frames* | *Ground truth* | *2-layer LSTM prediction* | *1-layer ConvLSTM prediction* |  *2-layer ConvLSTM prediction* | *3-layer ConvLSTM prediction* |
|<img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/1layer_conv_6427.gif" width="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/2layer_conv_6427.gif" width="200" /> | <img src="https://github.com/mingkuan94/Thesis_ConvLSTM/blob/master/gifs_and_plots/3layer_conv_6427.gif" width="200" /> | 
| *1-layer ConvLSTM prediction* |  *2-layer ConvLSTM prediction* | *3-layer ConvLSTM prediction* |


