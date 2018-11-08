# data-mining-proj
Group 31, Chuan De Sheng | Jordan Lam| Nathanael S Raj | Ng Tze Yang | Phang Jun Yu | Sarah Ye

### Instructions to Set Up

1. Make sure you have Anaconda  (latest) installed
2. For existing user, run `conda update conda`, then `conda update --all` to make sure the latest packages are installed
2. On your conda prompt, type: `conda create --name mining --file dependencies.txt `. If you desire to run python3.6, use: `conda create --name mining python=3.6 --file dependencies.txt `
3. Navigate to the project directory, and activate the environment
   For Windows: `activate mining`
   For Mac/Linux: `source activate mining`
4. Run `python main.py` to run all classifiers. Results are printed onto terminals and graphs are produced.



### Good Project Practices
1. In the main code (ie main.py) there is a sandbox area for you to write the code that does your part.
2. After writing that, copy and paste the code as a function into your own python module.
3. After that, in main.py, import the python module function and store the output in a variable so that the next person down the pipeline can easily access your data if needed.
4. If you need an example, you can look at the `importcsv.py` and `processdata.py` packages, and how I implemented those functions as a one liner in the main.py code
5. Pull before pushing, or create your own branch to ensure no code clashes exist! It is a pain in the ass to resolve merge errors.
6. If you ever import a new dependency, remember to import it using conda e.g. `conda install PACKAGE_NAME`. After that, update the dependency using `conda list --explicit > dependencies.txt`
7. If there is a dependency change, key in the command `conda install --yes --file dependencies.txt` while in the mining environment


Have a good time :)


### Notes on creating machine learning functions
1. For each classifier you are creating, e.g. svm, random forest etc., please create a predict function that takes in testX and returns the corresponding predictions
2. For example, create a function `bayesPredictions = bayesian.naiveBayes(testX, testY, trainX, trainY)` where `bayesPredictions` will represent the classifications of testX.
3. Place this function in main.py
4. You can look at the current code in main.py to get an idea
