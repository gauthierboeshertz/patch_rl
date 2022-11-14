import torch
import matplotlib.pyplot as plt

def patches_to_image(patches, patch_size, image_size):
    # transform batch of patches to image for square images and square patches
    image = torch.zeros((patches.shape[0],patches.shape[2],image_size,image_size))
    num_patch_row = image_size // patch_size
    for i in range(num_patch_row):
        for j in range(num_patch_row):
            image[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = patches[:,i*num_patch_row+j]
    return image

def image_to_patches(image,patch_size,num_patches_sqrt=None):
    image_patches = []
    for i in range(num_patches_sqrt):
        for j in range(num_patches_sqrt):
            image_patches.append(image[:,i*patch_size[0]:(i+1)*patch_size[0],j*patch_size[1]:(j+1)*patch_size[1]])
    return torch.stack(image_patches)

def plot_image_patches(image,patch_size,num_patches_sqrt):
    """ 
    Divide the image into patches of size patch_size and plot them
    """
    img_patches = image_to_patches(image,patch_size,num_patches_sqrt).byte()
    img_patches = img_patches.permute(0,2,3,1)
    fig = plt.figure(figsize=(9, 13))
    columns = num_patches_sqrt
    rows = num_patches_sqrt
    num_img_seq = img_patches.shape[-1]//3
    # ax enables access to manipulate each of subplots
    ax = []

    print(img_patches.shape)
    for i in range(columns*rows):
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title("patch:"+str(i))  # set title
        if num_img_seq == 1:
            plt.imshow(img_patches[i])
        else:
            for t in range(num_img_seq):
                plt.imshow(img_patches[i,:,:,t*3:(t+1)*3],alpha=0.2*(t+1))
            
    plt.show()
    return img_patches
