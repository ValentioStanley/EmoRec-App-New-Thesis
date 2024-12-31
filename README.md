Hello, welcome to my source code!

I created a source code for my thesis.

I will give you a guide how to use this application.

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
4. Write "pip install flask torch transformers", if you got error that program don't find torch transformer in the environment or something like that, then you need to open visual studio code or python development tools and install it on the cell.
5. For install libraries to meet preprocess requirement, please visit at Install-Lib.ipynb from Library_manual folder.

Third: Soon....