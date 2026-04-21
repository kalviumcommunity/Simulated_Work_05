# src/predict.py

def predict(model, pipeline, new_data):
    """Make predictions"""
    
    processed = pipeline.transform(new_data)
    return model.predict(processed)