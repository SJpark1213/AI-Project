from torchvision.models import wide_resnet50_2
import matplotlib.pyplot as plt
from torchvision import transforms

'model with no pretrain declaration'
model = wide_resnet50_2()

"model's first kernel declaration"

for kernel in model.parameters():
    kernel = kernel.data.cpu()
    break

"make grid 8x8"
plt.subplot(8, 8, len(kernel))
"Tensor to Pilimage"
transform = transforms.ToPILImage()

for i in range(64):
    plt.subplot(8,8,i+1)
    image = transform(kernel[i])
    plt.imshow(image)

plt.show()


'model with pretrain declaration'
model = wide_resnet50_2(pretrained=True)

"model's first kernel declaration"
for kernel in model.parameters():
    kernel = kernel.data.cpu()
    break
"make grid 8x8"
plt.subplot(8, 8, len(kernel))
"Tensor to Pilimage"
transform = transforms.ToPILImage()

for i in range(64):
    plt.subplot(8,8,i+1)
    image = transform(kernel[i])
    plt.imshow(image)

plt.show()