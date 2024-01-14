import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

st.title('Radiography Images Classification')
st.write('The [Github Repo](https://github.com/mohamedelhacen/radiography-images-classification)')
classes = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']
s = ''
for i in classes:
	s += '- ' + i + '\n'
st.markdown("This model can classify an __Xray__ image into one of these classes")
st.markdown(s)

st.write('#### Upload an Xray Image and press the **Classify** button')
uploaded_img = st.file_uploader('', type=['jpg', 'jpeg', 'png'])
if uploaded_img:
	col1, col2 = st.columns([1, 1])
	image = Image.open(uploaded_img).convert('RGB')
	with col2:
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
			probs = torch.softmax(outputs, 1)
			st.write(probs)
			st.write(f"This image has **{classes[pred.item()]}** with {round(probs[0][pred.item()].item()*100, 2)}% certainty.")
	with col1:
		st.image(image)

st.write('# Information about the data')
st.write("The dataset link [Kaggle](https://www.kaggle.com/datasets/preetviradiya/covid19-radiography-dataset)")
st.write("### Samples")
st.image('sample.png')
st.write('### The images distribution per class')
st.image('distribution.png')
