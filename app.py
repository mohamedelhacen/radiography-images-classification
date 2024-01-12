import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

title = st.title('Radiology classification')

classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
uploaded_img = st.file_uploader('', type=['jpg', 'jpeg', 'png'])
if uploaded_img:
	col1, col2 = st.columns([1, 1])
	image = Image.open(uploaded_img).convert('RGB')
	
	if st.button('Classify'):
		transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.PILToTensor()
								  ])
		img_tensor = transform(image)
		img_tensor = img_tensor.unsqueeze(0)
		model = torch.load('model.pt')
		with torch.no_grad():
			outputs = model(img_tensor.float())
		_, pred = torch.max(outputs, 1)
		with col2:
			st.write(f"This image has **{classes[pred.item()].replace('_', ' ')}**")
	with col1:
		st.image(image)

st.write('# Information about the data')
st.write('The data distribution')
st.image('distribution.png')