
# Define your local file path and remote 
local_output_file_path="/home/sureshp/Desktop/Landslide/Suresh/mri_outputs/"
remote_username="sureshp"
remote_host="login-mri.research.mtu.edu"
remote_output_file_path="Landslide/Suresh/outputs/"
remote_python_file='Landslide/Suresh/mycode.py'
remote_script_file='Landslide/Suresh/run.sh'


# download python script and batch script
scp "$remote_username@$remote_host:$remote_python_file" "$local_output_file_path"
#scp "$remote_username@$remote_host:$remote_script_file" "$local_output_file_path"

# Download the output folder file using scp
# scp -r "$remote_username@$remote_host:$remote_output_file_path" "$local_output_file_path"


