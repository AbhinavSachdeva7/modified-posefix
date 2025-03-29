
import streamlit as st
import numpy as np
import torch

import text2pose.config as config
from human_body_prior.body_model.body_model import BodyModel
import text2pose.utils as utils
import text2pose.data as data
import text2pose.utils_visu as utils_visu
from text2pose.generative_modifier.evaluate_generative_modifier import load_model
import roma


DEVICE = 'cpu'
nb_cols = 4 # for a nice visualization
margin_img = 80

def rotvec_to_eulerangles(x):
	x_rotmat = roma.rotvec_to_rotmat(x)
	thetax = torch.atan2(x_rotmat[:,2,1], x_rotmat[:,2,2])
	thetay = torch.atan2(-x_rotmat[:,2,0], torch.sqrt(x_rotmat[:,2,1]**2+x_rotmat[:,2,2]**2))
	thetaz = torch.atan2(x_rotmat[:,1,0], x_rotmat[:,0,0])
	return thetax, thetay, thetaz

def eulerangles_to_rotmat(thetax, thetay, thetaz):
	N = thetax.numel()
	# rotx
	rotx = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	roty = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotz = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotx[:,1,1] = torch.cos(thetax)
	rotx[:,2,2] = torch.cos(thetax)
	rotx[:,1,2] = -torch.sin(thetax)
	rotx[:,2,1] = torch.sin(thetax)
	roty[:,0,0] = torch.cos(thetay)
	roty[:,2,2] = torch.cos(thetay)
	roty[:,0,2] = torch.sin(thetay)
	roty[:,2,0] = -torch.sin(thetay)
	rotz[:,0,0] = torch.cos(thetaz)
	rotz[:,1,1] = torch.cos(thetaz)
	rotz[:,0,1] = -torch.sin(thetaz)
	rotz[:,1,0] = torch.sin(thetaz)
	rotmat = torch.einsum('bij,bjk->bik', rotz, torch.einsum('bij,bjk->bik', roty, rotx))
	return rotmat

def eulerangles_to_rotvec(thetax, thetay, thetaz):
	rotmat = eulerangles_to_rotmat(thetax, thetay, thetaz)
	return roma.rotmat_to_rotvec(rotmat)




def setup_body_model():
    body_model = BodyModel(model_type = config.POSE_FORMAT,
                       bm_fname = config.NEUTRAL_BM,
                       num_betas = config.n_betas)
    body_model.eval()
    body_model.to(DEVICE)
    return body_model



def process_img(img):
    return img[margin_img:-margin_img,margin_img:-margin_img]



model_path = 'Path of the model weights'
body_model = setup_body_model()
model, _ = load_model(model_path, device='cpu')




# --- layout
st.markdown("""
			<style>
			.smallfont {
				font-size:10px !important;
			}
			</style>
			""", unsafe_allow_html=True)




#  this pose works with an npy file, you can convert it for any data
def load_and_process_pose(npz_file, index=0, rotation_1=False):
	"""Load and process a pose from npy file"""
	
	pose_data = np.load(npz_file)
	# Assuming your npz contains SMPL-H pose parameters
	# pose = pose_data['poses']  # Should be shape (52, 3) or similar
	# pose = torch.from_numpy(pose).float()
	pose = torch.from_numpy(pose_data).float()
	pose = pose[index]

	
	if rotation_1:
		pose = pose.reshape(-1, 3)
		pose = torch.as_tensor(pose).to(dtype=torch.float32)

		initial_rotation = pose[:1,:].clone()
		thetax, thetay, thetaz = rotvec_to_eulerangles( initial_rotation )
		zeros = torch.zeros_like(thetaz)
		pose[0:1,:] = eulerangles_to_rotvec(thetax, thetay, zeros)
		
		
		pose = pose.reshape(1, -1)
	
	return pose














st.write(f"**Query data:**")
cols_input = st.columns([1,1,2])

st.write("**Select Subject Index:**")
sub1 = st.number_input("Subject 1", min_value=1, max_value=10, value=1)
sub2 = st.number_input("Subject 2", min_value=1, max_value=10, value=7)
pose_A_data_path = f'Path for poses(current pose)'
pose_B_data_path = f'Path for poses(target pose)'


st.write("**Select Pose Index:**")
index = st.number_input("Index", min_value=1, max_value=44, value=31) - 1

st.write("**Select Rotation:**")
rotation1 = st.checkbox("Rotation for subject 1")
rotation2 = st.checkbox("Rotation for subject 2")






# pose_A_data = data.T_POSE.view(1, -1)

pose_A_data = load_and_process_pose(pose_A_data_path, index=index, rotation_1=rotation1).view(1, -1)
pose_A_img = utils_visu.image_from_pose_data(pose_A_data, body_model, color="grey", add_ground_plane=True)

pose_A_img = process_img(pose_A_img[0])




pose_B_data = load_and_process_pose(pose_B_data_path, index=index, rotation_1=rotation2).view(1, -1)
pose_B_img = utils_visu.image_from_pose_data(pose_B_data, body_model, color="purple", add_ground_plane=True)
pose_B_img = process_img(pose_B_img[0])



cols_input[0].image(pose_A_img, caption="Current Pose" )
cols_input[1].image(pose_B_img, caption="Target Pose")
cols_input[2].write("_(Not annotated.)_")



st.markdown("""---""")
st.write("**Text generation:**")



models = []
models.append(model)


for i, model in enumerate(models):

	with torch.no_grad():
		texts, scores = model.generate_text(pose_A_data.view(1, -1, 3), pose_B_data.view(1, -1, 3)) # (1, njoints, 3)
		st.write(texts[0])
		st.markdown("""---""")
		st.write(f"_Results obtained with model: {model_path}_")