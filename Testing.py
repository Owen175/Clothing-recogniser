import training as t
import numpy as np
import matplotlib as plt
AI = t.model()

predictions = AI.model.predict(test_images)
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

num = int(input("Pick a number : "))

place = np.argmax(predictions[num])
print("Expected: ", class_names[test_labels[num]])

plt.figure()
plt.imshow(test_images[num], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

print("Guess: ", class_names[place])
