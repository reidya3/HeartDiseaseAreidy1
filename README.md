<h1 align="center">
  <img alt="Heart Disease infographic" src="./book/images/heart_disease.jpeg" height="115px" />
  <br/>
  Heart Disease Analysis & Prediction
</h1>
<h3 align="center">
  Anthony Reidy
  <br/><br/><br/>
</h3>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Development](#development)
- [Usage](#usage)
- [Web App](#web-app)


## Development
It is good practice to develop in a virtual environment. Note, this jupyter book was written using `python 3.7` and on the `MAC` operating system (OS). As such, all commands are setup for this installation and may not work for other OS's. To create a virtual environment called `venv`, execute:
```bash
python3 -m venv venv
```
To activate it, execute
```bash
source venv/bin/activate
```

- Execute `pip install -r requirements.txt` to install requirements for development.

## Usage
To build the jupyter book as HTML, please execute `jupyter-book build --all book/`. 

We use latex to build a PDF of our book for this investigation. For MacOS, you may want to install [MacTeX](https://tug.org/mactex/). Alternatively you may also use [TeX Live](https://www.tug.org/texlive/quickinstall.html).


Next, to build a PDF of the project, please use the following command `jupyter-book build  book/ --builder pdflatex`. **Note**, you will have to build a html version first. 

For further information on how to accomplish this on other OS's, please [click here](https://jupyterbook.org/advanced/pdf.html?highlight=build%20pdf). The **PDF file** can be found in the [docs](/docs) folder.


## Web App
Details of the web app can be found in the [app directory](app).