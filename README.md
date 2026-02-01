<h1>Video Instructions</h1>
https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=fcf9a70d-8eb1-4a40-8939-b3e4016f71d3

<h1>Written Instructions</h1>
<li>Ensure you have python installed by visiting the Official python website https://www.python.org/downloads/ and downloading atleast 3.14
Run the installer and check all the default options.
  </li>
<li>Ensure you have git installed by visiting the official git website https://git-scm.com/install/windows and downloading the installer
Run the installer and accept all the default settings.</li>

<li>Ensure you have mini conda installed by visiting the official Anaconda website. https://www.anaconda.com/download
Run the installer and accept all default settings.</li>
<li>Pull down the application from github by opening Anaconda terminal  and entering in git clone https://github.com/jwelt94/CatVsDogClassifier.git  note this may take a while there are quite a few images
<li>Change directories into the repo via cd CatVsDogClassifier
<li>Run the following command  conda create --name catVsDog python=3.13 pandas numpy pandas matplotlib tensorflow  seaborn scikit-learn opencv
<li>Activate the project by running the command and then conda activate catVsDog
<li>Install all dependencies by running conda install notebook numpy pandas matplotlib tensorflow  seaborn scikit-learn opencv
<li>Run the following commandpip install -U keras tensorflow
<li>Open a file explorer and navigate to the CatVsDogClassfier repo, and extract the model directly into the folder by right clicking CatsOrDogsModelSlim clicking extract all and ensuring the model is extracted directly into the CatsVsDogsClassifier directory.
<li>Go back to anaconda and Start up the jupyter notebook by running the following command jupyter notebook 
You may be asked how you want to open up the file. Choose your favorite web browser.
<li>Open up CatVsDogClassifier.ipynb.
<li>If you’d like to retrain the model run all stages by entering each block and pressing shift enter
<li>If you want to use the pretrained model, skip the training blocks. I have labeled these with comment for your situational awareness.
<li>Once you get to the last block and run it, a window with our GUI will open up. 
<li>You can upload any image of your liking by clicking the ‘Upload Image’ Button. The model will make a prediction on your image for you.
<li>If you want to see the dashboard with statistics on the models performance on our withheld training set, press the Evaluate Model button.
