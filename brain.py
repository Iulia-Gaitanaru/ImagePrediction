from imageai.Prediction import ImagePrediction
import os
# Grab the current directory
execution_path = os.getcwd()

prediction = ImagePrediction()
# Decide the model (SqueezeNet)
prediction.setModelTypeAsMobileNetV2()
# Model Path
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
# Load the model
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "godzilla.jpg"), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)