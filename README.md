# WebQMon.ai
Training datasets and algorithms of the WebQMon.ai
A recent study has found that interactive HTTP traffic once again dominated residential broadband Internet traffic, accounting for more than 50% of the traffic and it gradually becomes the de facto narrow waist of the Internet. People often visit varieties of websites at work or in their precious leisure time, including search engines, video sites and social networking sites. Whether the websites can load successfully whthin a short time is crucial before they enjoy the web services. And it is predictable that even tiny network delay of page load can ruin the user experience.

There are some key findings released by Akamai Technologies, Inc. in 2009 concerning about the correlation of e-commerce website performance and an online shopper's behavior. The direct conclusion drawn from the experiment with 1,048 online shoppers is that two seconds is a new threshold after which the consumers would become impatient if the pages is still not loaded and 40% of consumers will wait no longer than three seconds before they abandon the site. While waiting for a page to load, shoppers often get more distracted. 14% of them will shop at another site if it loads faster than the current site, and 23% will stop shopping or leave their computer. Additional research indicates that there is a negative feedback between slow page loading and consumers' loyalty to an e-commerce site, especially for high spenders. Up to 79% of online shoppers who experience a dissatisfying visit would never visit it again while 27% of them wouldn't visit its physical store neither. Therefore, the slow page loading will result in the loss of ICPs (Internet Content Providers) and cause inconvenience to the users. 

Quality of Service (QoS) is used widely in measuring the network's capability to provide the best service in a given network traffic and helps the ICT (Information and Communication Technology) engineers develop their products and provide better services. However, the relation between user experience and QoS is not a simply positive correlation, which means a high QoS does not necessarily lead to a good user experience. For reflecting user experience more accurate, the concept of Quality of Experience (QoE) is concerned as an alternative of QoS  by the ICT industry and ISPs in recent years. Because QoE successfully capture the feelings and needs of the consumer. QoE basically pays attention to how a user assesses and evaluates a service and measure how the service will affect the transaction amount from the perspective of the user experience. 

Over all, QoE will be increasingly important in measuring user experience in the future. The ISPs and equipment vendors can leverage the traffic through devices and a hypothetical QoE prediction model to estimate the current user experience anywhere in the network. By analyzing a series of QoE, they can refine the network equipment or connection timely or give feedbacks and recommendations to the website owner, thereby retrieving the losses caused by network delay of page load. Unfortunately, it is very difficult to assess the QoE in real-time on a running network. Monitoring network-level performance criteria is easier and more usual. But the problem is how to correlate these network-level QoS to the QoE perceived by the users.

Web-browsing QoE mainly depends on the above-the-fold time (AFT), that is, the loading time of the content that the display can directly show. Generally, the longer the AFT is, the worse the QoE will be. We can divide AFT into multiple intervals, and each interval corresponds to a certain QoE. For example, if the AFT is smaller than one second, the user experience will be good; if the AFT is greater than one second and smaller than five second, the user experience will get worse; if the AFT is greater than five second, the user experience will be terrible. Hence, we can evaluate web-browsing QoE by predicting the AFT.

We use traffic patterns to represent the TCP stream appearing when the user visits a website, which we believe vary with the change of the network condition. Under such a circumstance, we present a data-driven system, WebQMon.ai, which can predict the QoE with different AFT by distinguishing different forms of traffic patterns. Under this architecture, we propose five supervised learning-based methods to classify different forms of traffic patterns. The collected traffic patterns are labeled by the estimated AFT and then used as the training data. We train our machine learning models on the labeled data by continuously reducing the difference between the predicted and actual values of the model. Furthermore, WebQMon.ai works efficiently and it can predict AFT in a very short time. Specifically, more than two thousand samples can be predicted in less than a second. Moreover, when we predicted 2,400 unknown samples through our model, there is no prediction error and the prediction accuracy reaches 100%. 

# Algorithm
Algorithm of different models.
## LSTM.py
Training and prediction algorithm of the LSTM model.
## R-LSTM.py
Training and prediction algorithm of the R-LSTM model.
## Slice.py
Training and prediction algorithm of the NN model.
## NN.py
Training and prediction algorithm of the NN model.
## Combine.py
Training and prediction algorithm of the Combine model.
## Multi-LSTM.py
Training and prediction algorithm of the LSTM model for the ternary classification problem.
## Multi-R-LSTM.py
Training and prediction algorithm of the R-LSTM model for the ternary classification problem.
## Multi-Slice.py
Training and prediction algorithm of the Slice model for the ternary classification problem.
## Multi-NN.py
Training and prediction algorithm of the NN model for the ternary classification problem.
## Multi-Combine.py
Training and prediction algorithm of the Combine model for the ternary classification problem.	
## train_data_merge.py
For reading all files in the training data folder.
# TrainingData
Training datasets for different websites.
## amazon
Training datasets for Amazon.
## sina
Training datasets for Sina.
## youku
Training datasets for youku.
## sina-Ternary
Training datasets for sina for ternary classification problem.
# my_combine_model
The parameters of the trained Combine model. You can load the parameters from this folder and predict the AFT directly.
# my_lstm_model
The parameters of the trained LSTM and R-LSTM model. You can load the parameters from this folder and predict the AFT directly.
# my_nn_model
The parameters of the trained NN model. You can load the parameters from this folder and predict the AFT directly.
# my_slice_model
The parameters of the trained Slice model. You can load the parameters from this folder and predict the AFT directly.
# How to run
First, you need to put all the files in the “Algorithm” folder in the same directory as the other folders.
Then, if you installed the dependencies successfully, you can train or test each model by each model.py.
