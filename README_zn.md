To run in jupter notebook on Wendian

## For dmg framework:
1. I have moved my environments saved on wendian due to space limit, therefore make sure you activate the right environment from the right location, which is at /beegfs/scratch/zhennanshi/conda_envs/envs/, use:

conda activate /beegfs/scratch/zhennanshi/conda_envs/envs/dmg312

2. Now go to your delta model location, which is at /beegfs/scratch/zhennanshi/DM/generic_deltamodel, then run the toml file at where the toml is /beegfs/scratch/zhennanshi/DM/generic_deltamodel/pyproject.toml, in order to install all the dependencies:

python -m pip install -e .  ### Do not use pip install -e .###

3. Create the kernel for the jupter notebook to run, skip if the kenerl is already created!

python -m pip install ipykernel
python -m ipykernel install --user --name dmg312 --display-name "Python (dmg312_renewed)"

4. Now you can select on the Jupter notebook: Kenrnel --> Reconnect to (select the one you created), skip if you used the same kernel as last time

## For hydrodl2:
Since you also used hydrodl2, which is at https://github.com/mhpi/hydrodl2, make sure it is up-to-date:

0. Back-up files you modified
download a version locally

1. You want to keep the HBV class that you modified:
git stash

2. You want to pull
git pull

3. Check what is changed
git stash pop

4. Go to where the toml file is at in hydrodl2, then run:
python -m pip install -e .

## For GEFS Data

1. Make sure it is at /u/st/dr/awwood/aw-ciroh-proj/projects/dl_da/daymet-gefs-camels-gII/, eg. /u/st/dr/awwood/aw-ciroh-proj/projects/dl_da/daymet-gefs-camels-gII/ens01 for ensemble 1

2. 



Some useful commands:
1. search a string for debugging purpose

grep -R "your_string" folder_name

2. check the training progress by running the file or in terminal
!tail -n 20 -f train_progress.log
module load apps/python3
conda activate hydro

3. Copy the file on wendian, eg. /beegfs/scratch/zhennanshi/DM/ and save it locally
scp -r zhennanshi@wendian.mines.edu:/beegfs/scratch/zhennanshi/DM/ ~/Downloads/