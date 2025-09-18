# pyComBat for Normalization

# Install pyComBat package
pip install combat

#upgrading pyComBat to its latest version 
pip install combat --upgrade

#Running pyComBat
#df: The expression matrix as a dataframe and batch used is 5.
from combat.pycombat import pycombat
data_corrected = pycombat(df,5)
