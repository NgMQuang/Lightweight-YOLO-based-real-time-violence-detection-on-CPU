Those codes actually cannot run due to some problems, but you can look for the idea and implementation yourself.

Note:

Training data: All from RWF-2000 train dataset, which cut for meaningful interactions for Fights slit uniformly for NonFights' videos. Total there about ~12000 images for training and ~3000 images for validating for YOLO (No testing cause not my problem), 1503 videos to train and 398 to validate. The RWF validating folder is kept for testing.

The YOLO training code is in violence_yolo.ipynb, the temporal classifier training is lost, but you can find the architecture in TemporalClassifier.ipynb

The execution model is FP32, just convert to ONNX type. No optimization cause I dont find it necessary for my research scope.

There are some data processing, debugging, cleaning, … scripts I dont show here cause lost. But testing data is preserved and keep untouched for formal test. I use a sliding window to "detect" the highest fight probability (from Temporal Classifier).

Lessons:

- Do not use attention mechanism for this kind of application. It's easy to get overfitting and too heavy to operate.
- Hook is one of the best way to study behaviors of YOLO, that is the inspiration for this model
- Do not train on your laptop (?), use a server so the server get the pain.



