Hello, Welcome to my source code!

EmoRec = Emotion Recognition
I created a source code for my thesis.

I will give you a guide how to use this application for only windows.
If you are using Mac or Linux, please explore the browser. I'm sure the step by step should be almost similar to this guide.

First: Installing Python Environment
1. You need to install a python file, please visit at https://www.python.org/downloads/
2. Select python version 3.12 or lower because transformers and torch library don't support the latest version.
3. Open the download folder.
4. Open the python which you downloaded, select all of checkbox "Add python.exe to PATH" and "Use admin privilieges when installing py.exe", and then select "Install Now" unless select "Customize Installation" if you want a custom.
5. Waiting until the Installation is done, select disable path length limit (optional).
6. Open the command prompt and type "python --version", if the command displays python version then congratulations! Otherwise, please check at environment variables.

Second: Installing Library
1. Create a new workspace folder. After that, click the path of workspace folder, write "cmd", and then it will displays command prompt automatically.
2. Write "python -m venv env" to create a new package.
3. Write "env\Scripts\activate" (for windows) to activate the package where you created. 
4. Write "pip install flask torch transformers", if you got error that program doesn't find torch transformer in the environment or something like that, then you need to open visual studio code or python development tools and install the pip on the cell.
5. For install libraries, please visit at Install-Lib.ipynb from Library_manual folder to meet the preprocess and data mining requirements

Third: Saving model on the python application
1. Select IndoBert-4 or Logistic Regression File with python extension. 
2. Change the location directory code according to your location directory in the bottom from training code (optional)

Machine Learning:
import pickle
pickle.dump(lr, open('model/machine_learning/lr_old/lr.pkl', 'wb'))

Deep Learning:
model.save_pretrained("model")
tokenizer.save_pretrained("model")

3. Click the run all code until the model being saved in your location directory.

For your Information:
The save pretrained needs some your NVIDIA local GPU to run bert file so you have to download some NVIDIA GPU resource. Although, the NVIDIA GPU Resources have big size, this may impact your local storage.
If you want to save local storage capacity, please visit at jupyter notebook (https://colab.google/ or https://www.kaggle.com/code or any notebook you prefer)
After visitting them, please follow this third step until save the model into your notebook directory. 
Please follow the model saved on the notebook directory, then download the model into your workspace folder.
For example:
![Save Model](guide-image/image.png)

Fourth: Launching application
1. Select IndoBert (top) or Logistic Regression (bottom) snippet code, then un-comment the snippet code selected.
2. Click the run all code until the application being launched or write "python app.py" at terminal vscode.
3. Open the browser and type "http://127.0.0.1:5000 or follow the instruction at terminal vscode.
4. Input text or sentence review to the application and it will display the emotion recognition result.
5. If it is working, then congratulations!

If you need a hand, please don't hesitate to hit me up through discord: iabyes