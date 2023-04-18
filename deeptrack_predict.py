#%%
import numpy as np
import matplotlib.pyplot as plt
import deeptrack as dt
import matplotlib.pyplot as plt
import pims
import skimage.io
import pandas as pd


loss = dt.losses.flatten(
    dt.losses.weighted_crossentropy((10, 1))
)
metric = dt.losses.flatten(
    dt.losses.weighted_crossentropy((1, 1))
)

model_path = "/mnt/f/Seafile/我的资料库/文档/PHYS 350/Winter/models/"
weights = [f"model_z={z}" for z in [-200,-150,-100,-50,0,50,100,150,200]]
models = {}
for weight in weights:
    model = dt.models.UNet(
        (None, None, 1), 
        conv_layers_dimensions=[16, 32, 64],
        base_conv_layers_dimensions=[128, 128], 
        loss=loss,
        metrics=[metric],
        output_activation="sigmoid"
    )
    model.model.load_weights(model_path+weight+'.h5')
    models[weight] = model

#12.9 trapped: x_lim = (520,550) y_lim = (850,890)

video_path = "/mnt/f/Seafile/我的资料库/文档/PHYS 350/Winter/Cut Videos/"
video_path += "72.9x200,t=0 30fps.mov"
plot_example_fig = False

x_lim = (330,370) #location of the partical
y_lim = (630,670)
if plot_example_fig:
    read_size = 30
else:
    read_size = 1000
min_image_size = 256

if not plot_example_fig:
    image_size = min_image_size*(1+max(x_lim[1]-x_lim[0],y_lim[1]-y_lim[0])//min_image_size)

    x_center = (x_lim[1]+x_lim[0])//2
    x_splices = (x_center-image_size//2, x_center+image_size//2)
    x_lim = (x_lim[0]-x_splices[0],x_lim[1]-x_splices[0])

    y_center = (y_lim[1]+y_lim[0])//2
    y_splices = (y_center-image_size//2, y_center+image_size//2)
    y_lim = (y_lim[0]-y_splices[0],y_lim[1]-y_splices[0])

    print("x_lim:",x_lim)
    print("y_lim:",y_lim)

images = pims.PyAVReaderTimed(video_path)
frame_rate = images.frame_rate
images = pims.as_grey(images)
print(f'There are {len(images)} frames. frame rate is {frame_rate}')
data_collected = {weight:np.zeros((len(images),2)) for weight in weights }


for n in range((len(images)-1)//read_size+1):
    print(n*read_size,min((n+1)*read_size,len(images)))
    images_read = None
    images_read = np.array(images[n*read_size:min((n+1)*read_size,len(images))])
    if not plot_example_fig:
        images_read = images_read[:, x_splices[0]:x_splices[1], y_splices[0]:y_splices[1] ]
    print(images_read.shape)
    images_read = np.expand_dims(images_read, axis=-1)
    images_read = dt.NormalizeMinMax(0, 1).resolve(images_read)
    for weight in weights:
        predictions = models[weight].predict(np.array(images_read), batch_size=1)
        for i, (prediction, image) in enumerate(zip(predictions, images_read)):
            mask = prediction[:,:,0] == 1

            cs = np.array(skimage.measure.regionprops(skimage.measure.label(mask)))
            
            cs = np.array([c["Centroid"] for c in cs])

            if plot_example_fig:
                print("all:",cs)
                plt.subplot(1,2,1)
                plt.imshow(image, cmap="gray")  
                #plt.axis("off")
                plt.axis([0, image.shape[1], 0, image.shape[0]])
                
                plt.subplot(1,2,2)
                plt.imshow(image, cmap="gray")
                plt.scatter(cs[:, 1], cs[:, 0], 100, facecolors="none", edgecolors="w")
                #plt.axis("off")
                plt.axis([0, image.shape[1], 0, image.shape[0]])
                
                #plt.savefig(f"predict/{i}.jpg",dpi=400)
                plt.draw()
                plt.pause(0.01)
                plt.cla()

            try:
                cs = cs[(x_lim[0]<cs[:,0]) & (cs[:,0]<x_lim[1]) & (y_lim[0]<cs[:,1]) & (cs[:,1]<y_lim[1])]
                if plot_example_fig:
                    print("filtered:",cs)
                if len(cs) == 1:
                    data_collected[weight][i+n*read_size] = cs[0]
                else:
                    data_collected[weight][i+n*read_size] = np.full(2,np.nan,dtype=float)
            except:
                data_collected[weight][i+n*read_size] = np.full(2,np.nan,dtype=float)

data = {"time (s)": np.arange(len(images))/frame_rate}
for weight in weights:
    data[f"x (pts)({weight})"] = data_collected[weight][:, 0]
    data[f"y (pts)({weight})"] = data_collected[weight][:, 1]


data = pd.DataFrame(data)
print(data)
data.to_csv(video_path[:-4]+"_new.csv",index=False)

for weight in weights:
    plt.subplot(1,2,1)
    plt.plot(np.arange(len(images))/frame_rate, data_collected[weight][:, 0])
    plt.subplot(1,2,2)
    plt.plot(np.arange(len(images))/frame_rate, data_collected[weight][:, 1])
    plt.title(weight)
    plt.show()

# %%
