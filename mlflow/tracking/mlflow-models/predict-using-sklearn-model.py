import mlflow
import numpy as np


# mlflow.set_tracking_uri("http://mlflow-tracking-server:5000")
logged_model = 'runs:/9656ead0db8e4696bdbaabf91353079a/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)
random_data = np.random.randn(10, 87931)
prediction = loaded_model.predict(random_data)
print(prediction)
