# Necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# Load content and style images
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

content_image = load_image("content.jpeg")
style_image = load_image("pattern.jpeg")

# Define a function to display images
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Display the content and style images
plt.figure()
imshow(content_image, title="Content Image")
plt.figure()
imshow(style_image,title="Style Image")

#Load the VGG19 model
cnn = models.vgg19(pretrained=True).features.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()

# Define content and style loss classes
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self,x):
        self.loss = nn.functional.mse_loss(x,self.target)
        return x
    
class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss,self).__init__()
        self.target = self.gram_matrix(target).detach()
    
    def forward(self,x):
        self.loss = nn.functional.mse_loss(self.gram_matrix(x), self.target)
        return x

    def gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

def get_model_and_losses(cnn, style_img, content_img):
    # cnn = cnn.features.to(device).eval()

    content_layers=['conv_4']
    style_layers = ['conv_1','conv_2','con_3', 'conv_4','conv_5']

    content_losses =[]
    style_losses = []

    model=nn.Sequential()

    i=0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, content_losses, style_losses

model, content_losses, style_losses = get_model_and_losses(cnn, style_image, content_image)

input_image = content_image.clone()
optimizer = optim.LBFGS([input_image.requires_grad_()])

def run_style_transfer(model, content_losses, style_losses, input_image, num_steps=30, style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    print('Optimizing..')
    run=[0]
    while run[0] <= num_steps:

        def closure():
            input_image.data.clamp_(0,1)

            optimizer.zero_grad()
            model(input_image)
            style_score=0
            content_score=0

            for s1 in style_losses:
                style_score+=s1.loss
            for c1 in content_losses:
                content_score+=c1.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score+content_score
            loss.backward()

            run[0]+=1
            if run[0]%15==0:
                print("run {}:".format(run[0]))
                print('Style Loss: {:4f} Content Loss: {:4f}'.format(style_score.item(),content_score.item()))
                print()

            return style_score+content_score
        
        optimizer.step(closure)

    input_image.data.clamp_(0,1)

    return input_image

output = run_style_transfer(model, content_losses, style_losses, input_image)

output_image = output.cpu().clone()
output_image = output_image.squeeze(0)
output_image = transforms.ToPILImage()(output_image)
output_image.save("output_image.jpg")

plt.figure()
imshow(output, title ='Output Image')
plt.ioff()
plt.show()