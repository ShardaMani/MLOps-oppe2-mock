from locust import HttpUser, task, between

class PredictUser(HttpUser):
    wait_time = between(0.5, 1)

    @task
    def predict(self):
        payload = {
            "sno": 1,
            "age": 45,
            "gender": "male",
            "cp": 0,
            "trestbps": 120,
            "chol": 240,
            "fbs": 0,
            "restecg": 1,
            "thalach": 160,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 0,
            "thal": 3
        }
        self.client.post("/predict", json=payload)
