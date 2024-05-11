
# Define your local file path and remote destination on the cluster
remote_username="sureshp"
remote_host="login-mri.research.mtu.edu"
remote_destination_scripts="Landslide/Suresh/uploaded_scripts/"

local_python_file='scripts_to_upload/mycode.py'
local_script_file='scripts_to_upload/run.sh'

# Transfer the file using scp
# scp "$local_file_path" "$remote_username@$remote_host:$remote_destination"
scp "$local_python_file" "$remote_username@$remote_host:$remote_destination_scripts"
scp "$local_script_file" "$remote_username@$remote_host:$remote_destination_scripts"
