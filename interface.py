from tkinter import *
from tkinter.filedialog import askopenfilename
from os import getcwd
from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import models


# some styles
class ModernButton(Button):
	def __init__(self, master, **kw):
		Button.__init__(self,master=master,**kw)
		self.defaultBackground = self["background"]
		self.bind("<Enter>", self.on_enter)
		self.bind("<Leave>", self.on_leave)

	def on_enter(self, e):
		self['background'] = self['activebackground']

	def on_leave(self, e):
		self['background'] = self.defaultBackground

class StandardButton(ModernButton):
	def __init__(self, master, text, borderwidth=0, relief=SUNKEN, bg="#242020", fg="white", font='{Arial} 12',
		width=25, height=3, activebackground="#747474", **kw):
		ModernButton.__init__(self, master=master, borderwidth=borderwidth, text=text,
			relief=relief, bg=bg, fg=fg, font=font,
			width=width, height=height, activebackground=activebackground, **kw)

class StandardHeading(Label):
	def __init__(self, master, text, width=100, height=3, font="{Arial} 24 bold",**kw):
		Label.__init__(self, master=master, text=text, width=width, height=height, font=font, **kw)

class StandardLabel(Label):
	def __init__(self, master, text, width=100, height=3, font="{Arial} 12",**kw):
		Label.__init__(self, master=master, text=text, width=width, height=height, font=font, **kw)

class BufferLabel(Label):
	def __init__(self, master, text="        ", width=100, height=3, font="{Arial} 10",**kw):
		Label.__init__(self, master=master, text=text, width=width, height=height, font=font, **kw)

class StandardText(Text):
	def __init__(self, master, width=50, borderwidth=0, height=1, font="{Arial} 10", **kw):
		Text.__init__(self, master=master, width=width, height=height, borderwidth=borderwidth, font=font, **kw)

class StandardEntry(Entry):
	def __init__(self, master, width=50, borderwidth=0, font="{Arial} 10", **kw):
		Entry.__init__(self, master=master, width=width, borderwidth=borderwidth, font=font, **kw)

class Application:

	def __init__(self):

		self.model = models.load_model("cifar10_vgg19type_convolutionalNeuralNetwork.h5")

		self.root = Tk()
		self.root.geometry("850x650")
		self.root.title("Cifar - 10 CNN Interactive tool")
		self.root.resizable(0, 0)

		self.chosen_image = None

		heading = StandardHeading(self.root, "CIFAR-10").pack()
		info_label = StandardLabel(self.root,
			text="This interface allows you to interact with the Convolutional Neural Network trained on the Cifar-10 Dataset").pack()

		img_heading = StandardLabel(self.root, text="Currently chosen file: ").pack()
		self.current_display_path = StandardLabel(self.root, text="No image chosen")
		self.current_display_path.pack()

		file_choose_btn = StandardButton(self.root, text="Choose an Image", command=self.choose_file).pack()
		buffer_label = BufferLabel(self.root).pack()		
		predict_btn = StandardButton(self.root, text="Predict Image", command=self.predict).pack()

		self.root.mainloop()

	def choose_file(self):
		filename = askopenfilename(initialdir=getcwd(), title="Choose an Image", filetypes=[("JPG files", '*.jpg')])
		self.chosen_image = filename
		self.current_display_path.configure(text=filename)

	def error_window(self, text):
		error_window = Tk()
		error_window.title("Error")
		error_window.geometry("500x200")
		error = StandardLabel(error_window, text=text)

		error.pack()

		def destroy_window():
			error_window.destroy()

		btn = StandardButton(error_window, text="Okay", command=destroy_window).pack()
		buffer = BufferLabel(error_window).pack()
		error_window.mainloop()

	def predict(self):
		if self.chosen_image == None:
			self.error_window(text="No image has been chosen. Please choose an image.")
			return 0

		class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		IMG_PATH = self.chosen_image

		# load the image and convert into
		# numpy array
		img = Image.open(IMG_PATH)
		img = img.resize((32, 32))
		# img.show()
		# asarray() class is used to convert
		# PIL images into NumPy arrays
		arr = asarray(img)
		arrd = arr
		# print(arr)
		arr.shape = (32, 32, 3)
		arr = arr/255.0
		arr = np.expand_dims(arr, axis=0)

		# This image will be displayed
		img_display = mpimg.imread(IMG_PATH)

		# loading the model
		cifar10cnn = self.model
		prediction = cifar10cnn.predict([arr])
		plt.title("Prediction: " + class_names[np.argmax(prediction[0])])
		plt.imshow(img_display, cmap=plt.cm.binary)
		plt.show()


app = Application()