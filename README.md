# Plot Eraser
[Dash](https://github.com/plotly/dash)-based instrument for correcting data visually.
_____

### Use example

**Initialized Dash PlotEraser app**
![Initialized Dash PlotEraser app](images/virgin_ploteraser_frame.png "Initialized Dash PlotEraser app
")

**Selected with Lasso instrument area**

![Lasso instrument](images/plotly_instruments_pannel.png " Lasso instrument
")
![Selected with Lasso instrument area](images/lasso_ploteraser_frame.png "Selected with Lasso instrument area
")


**Erasing selected part**
![Erasing selected part](images/erased_ploteraser_frame.png "Erasing selected part
")

_____

### How to run

#### Option 1
Run with own python interpreter using `app/plot_eraser.py` script. See code for more info about CLI

Example: 
<br>`python3 run app/plot_eraser.py`
<br>`python3 run app.plot_eraser.py --source_file_path my_own_table.csv --source_file_extension csv`

Terminal will forward port to open in browser

#### Option 2

Use docker to create container
<br>`docker build -t smiling/ploteraser <path_to_repo_folder>`

Run created container
<br>`docker run -p 127.0.0.1:8050:8050 smiling/ploteraser`
<br>*Here `-p 127.0.0.1:8050:8050` means we forward local host port 8050 to the container's port 8050. This port is fixated both in code and Dockerfile*

App will be available in one's browser via `http://127.0.0.1:8050/`
