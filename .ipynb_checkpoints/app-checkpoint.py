{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cd087148-b74e-4020-a76a-279a9208e64a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5000\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "INFO:werkzeug: * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\mahes\\anaconda3\\Lib\\site-packages\\IPython\\core\\interactiveshell.py:3585: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "from flask import Flask, request, jsonify\n",
    "from tensorflow.keras.models import load_model\n",
    "import numpy as np\n",
    "import base64\n",
    "from PIL import Image\n",
    "import io\n",
    "\n",
    "app = Flask(__name__)\n",
    "model = load_model(\"mnist_model.h5\")\n",
    "\n",
    "@app.route(\"/predict\", methods=[\"POST\"])\n",
    "def predict():\n",
    "    data = request.json\n",
    "    if \"image\" not in data:\n",
    "        return jsonify({\"error\": \"No image provided\"}), 400\n",
    "\n",
    "    # Decode base64 image\n",
    "    image_data = base64.b64decode(data[\"image\"])\n",
    "    image = Image.open(io.BytesIO(image_data)).convert(\"L\").resize((28, 28))\n",
    "    image_array = np.array(image).reshape(1, 28, 28) / 255.0\n",
    "\n",
    "    prediction = model.predict(image_array)\n",
    "    predicted_class = np.argmax(prediction)\n",
    "\n",
    "    return jsonify({\"prediction\": int(predicted_class)})\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.run(debug=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e9b6677-7644-474c-b29c-f3f298c7d528",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
