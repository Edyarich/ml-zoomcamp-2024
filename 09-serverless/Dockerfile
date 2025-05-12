FROM agrigorev/model-2024-hairstyle:v3

# Update pip if needed
RUN pip install --upgrade pip

# Install specific numpy version
RUN pip install numpy==1.23.1

# Add the TFLite runtime wheel from the provided URL
ADD https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl /tmp/

# Install the TFLite runtime
RUN pip install /tmp/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

# Copy your lambda code file (assume it's named handler.py) into the image
COPY handler.py .

# Set the command for Lambda
CMD ["handler.handler"]
