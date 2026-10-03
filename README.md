CONVENTIONS OF CODE BASE

data: current data folder, will be populated over time
- molarityAmount_chemicalUsed_timeSeries
    - ilastik masks (probability masks)
    - raw tiff files

masking: this folder is for masking pipeline

scripts: main folder
MAIN RULE: attach a DATE to when you made it at the end... regardless of name! for example: ###_Oct-3.ipynb or ###_Oct-3.ipynb

- jupyter (mostly explorative and viewing scripts)
- python
    - patch_creation: for splitting stuff into patches
    - gridsearch folder: for making better masks and individually selecting mito
    - analysis: either logistical regression or later CNN if data allows

session_log: what was done for the day, to be made alongside git but with natural language because git gets messy# mitoEnd2End
