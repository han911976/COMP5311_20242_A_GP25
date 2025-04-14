
import matplotlib.pyplot as plt
import numpy as np

# loss = [0.4504, 0.2884, 0.2023, 0.1545, 0.1297, 0.1135, 0.0997, 0.0904, 0.0803, 0.0703, 0.0587, 0.0466, 0.0389, 0.0335, 0.0298, 0.0274, 0.0259, 0.0256, 0.0237, 0.0189]
# # epochs = range(1,len(loss)+1)
# epochs = range(0,len(loss))
# plt.plot(epochs, loss, "b-", label="Training Loss")
# plt.title("Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.xticks(epochs[::2])
# plt.legend()
# # plt.show()
# plt.savefig('test.png')


# loss_50_2 = [0.7956, 0.5212, 0.4264, 0.3657, 0.3196, 0.2805, 0.2450, 0.2126, 0.1857, 0.1632]
# loss_50_3 = [0.7408, 0.4221, 0.2911, 0.2159, 0.1715, 0.1431, 0.1215, 0.1037, 0.0888, 0.0758]
# loss_50_4 = [0.6672, 0.4276, 0.3086, 0.2358, 0.1853, 0.1498, 0.1237, 0.1040, 0.0887, 0.0760]
# loss_80_2 = [0.4294, 0.1737, 0.0864, 0.0517, 0.0349, 0.0262, 0.0202, 0.0163, 0.0131, 0.0107]
# loss_80_5 = [0.3192, 0.0915, 0.0412, 0.0219, 0.0165, 0.0163, 0.0138, 0.0116, 0.0099, 0.0086]
# epochs = range(0,10)

# plt.plot(epochs, loss_50_2, "r-", label="Training Loss [50,2]")
# plt.plot(epochs, loss_50_3, "g-", label="Training Loss [50,3]")
# plt.plot(epochs, loss_50_4, "b-", label="Training Loss [50,4]")
# plt.plot(epochs, loss_80_2, "m-", label="Training Loss [80,2]")
# plt.plot(epochs, loss_80_5, "y-", label="Training Loss [80,5]")
# plt.title("Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.xticks(epochs[::2])
# plt.legend()
# # plt.show()
# plt.savefig('training_loss_all.png')
# plt.clf()

plt.plot(15.3435, 0.939976, "x", label="[50,2]")
plt.plot(20.1140, 0.958133, "x", label="[50,3]")
plt.plot(25.5466, 0.961435, "x", label="[50,4]")
plt.plot(21.2147, 0.994448, "x", label="[80,2]")
plt.plot(47.9295, 0.994598, "x", label="[80,5]")
plt.title("Accuracy vs Time")
plt.xlabel("Training Time(s)")
plt.ylabel("Accuracy")
# plt.xticks(epochs[::2])
plt.yticks(np.arange(0.9,0.99,0.02))
plt.legend(loc='lower right')
# plt.show()
plt.savefig('time_acc.png')
