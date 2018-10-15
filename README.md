# data-mining-proj

### Instructions to Set Up

1. Make sure you have Anaconda  (latest) installed
2. For existing user, run `conda update conda`, then `conda update --all` to make sure the latest packages are installed
2. On your conda prompt, type: `conda create --name mining --file dependencies.txt `
3. Navigate to the project directory, and activate the environment
   For Windows: `activate mining`
   For Mac/Linux: `source activate mining`
4. Open your favourite file editor and start hacking :)

### Good Project Practices
1. In the main code (ie main.py) there is a sandbox area for you to write the code that does your part.
2. After writing that, copy and paste the code as a function into your own python module.
3. After that, in main.py, import the python module function and store the output in a variable so that the next person down the pipeline can easily access your data if needed.
4. If you need an example, you can look at the `importcsv.py` and `processdata.py` packages, and how I implemented those functions as a one liner in the main.py code
5. Pull before pushing, or create your own branch to ensure no code clashes exist! It is a pain in the ass to resolve merge errors.
6. If you ever import a new dependency, remember to import it using conda e.g. `conda install PACKAGE_NAME`. After that, update the dependency using `conda list --explicit > dependencies.txt`


Have a good time :)
