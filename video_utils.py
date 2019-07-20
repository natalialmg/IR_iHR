import numpy as np
import dlib
import glob
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def mask_generation_simple(video,land_drift, land_index=None):
    mask_L = np.zeros(video.shape[:-1])
    src = np.median(land_drift, axis=2)
    
    if land_index is not None:
        polygon =[(src[l,0],src[l,1]) for l in land_index]
        imgmask = Image.new('L', (video.shape[1],video.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon, outline=1, fill=1)
        mask_L = np.array(imgmask)
    else:
        ms= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,26,25,24,19,18,17]
        e1= [36,37,38,39,40,41]
        e2= [42,43,44,45,46,47]
        bc= [48,49,50,51,52,53,54,55,56,57,58,59]
        polygon_ms =[(src[l,0],src[l,1]) for l in ms]
        polygon_e1 =[(src[l,0],src[l,1]) for l in e1]
        polygon_e2 =[(src[l,0],src[l,1]) for l in e2]
        polygon_bc =[(src[l,0],src[l,1]) for l in bc]
        
        imgmask = Image.new('L', (video.shape[1],video.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_ms, outline=1, fill=1)
        mask_L = np.array(imgmask)
        
        #remove eye 1
        imgmask = Image.new('L', (video.shape[1],video.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_e1, outline=1, fill=1)
        mask_L -= np.array(imgmask)
        
        #remove eye 2
        imgmask = Image.new('L', (video.shape[1],video.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_e2, outline=1, fill=1)
        mask_L -= np.array(imgmask)
        
        #remove mouth
        imgmask = Image.new('L', (video.shape[1],video.shape[0]), 0)
        ImageDraw.Draw(imgmask).polygon(polygon_bc, outline=1, fill=1)
        mask_L -= np.array(imgmask)
    return mask_L

def get_region_values(video_t,pre_land):
    [nx,ny] = video_t.shape
    region = {}

    #Forehead
    region['LFH1'] = np.array([20,21,22,78,79,72,71,70]) - 1
    region['RFH1'] = np.array([78,23,24,25,75,74,73,79]) - 1
    region['LFH2'] = np.array([1,18,19,20,70,69]) - 1
    region['RFH2'] = np.array([17,77,76,75,25,26,27]) - 1

    #eyes region
    region['LE1'] = np.array([1,37,38,20,19,18])-1
    region['RE1'] = np.array([17,27,26,25,45,46])-1
    region['LE2'] = np.array([38,39,40,28,78,22,21,20])-1
    region['RE2'] = np.array([28,43,44,45,25,24,23,78])-1

    #nose region
    region['LN1'] = np.array([32,33,34,31,30,29,28,40])-1
    region['RN1'] = np.array([34,35,36,43,28,29,30,31])-1

    #cheeck region
    region['LCU1'] = np.array([81,32,40,41,42])-1
    region['LCU2'] = np.array([81,42,37,1,2,3])-1
    region['RCU1'] = np.array([36,83,47,48,43])-1
    region['RCU2'] = np.array([83,15,16,17,46,47])-1

    region['LCD1'] = np.array([81,32,49,80])-1
    region['LCD2'] = np.array([80,81,3,4])-1

    region['RCD1'] = np.array([36,83,82,55])-1
    region['RCD2'] = np.array([83,82,14,15])-1

    #mouth region
    region['LM'] = np.array([49,50,51,52,34,33,32])-1
    region['RM'] = np.array([52,53,54,55,36,35,34])-1

    #chin region
    region['LC1'] = np.array([4,5,6,60,49]) - 1
    region['RC1'] = np.array([55,14,13,12,56]) - 1

    region['LC2'] = np.array([6,7,8,59,60]) - 1
    region['RC2'] = np.array([12,56,57,10,11]) - 1
    region['CC'] = np.array([10,57,58,59,8,9]) - 1
    
    mask = np.zeros(video_t.shape)
    
    #Build Region Masks
    boosted_landmarks = np.array(pre_land[:,:])
    face_length = np.max(boosted_landmarks[:,1])-np.min(boosted_landmarks[:,1])
    idx_forehead = np.arange(18,27)

    aux_fh = np.array(boosted_landmarks[idx_forehead,:])
    aux_fh[:,1] = aux_fh[:,1] - face_length*0.25
        
    new_points = np.zeros([6,2])
    
    #between eyebrows
    new_points[0,:] = (boosted_landmarks[21,:] + boosted_landmarks[22,:])/2
    new_points[1,:] = (aux_fh[3,:] + aux_fh[4,:])/2
    
    #middle left cheek
    new_points[2,:] = (boosted_landmarks[48,:] + boosted_landmarks[3,:])/2
    new_points[3,:] = (boosted_landmarks[31,:] + boosted_landmarks[1,:])/2

    #middle left cheek
    new_points[4,:] = (boosted_landmarks[54,:] + boosted_landmarks[13,:])/2
    new_points[5,:] = (boosted_landmarks[35,:] + boosted_landmarks[15,:])/2

    boosted_landmarks = np.concatenate([boosted_landmarks,aux_fh,new_points],axis = 0)
    cont = 1
    for k in region.keys():
        mask_k = mask_generation_simple(video_t[:,:,np.newaxis],boosted_landmarks[:,:,np.newaxis], land_index=region[k])
        mask = np.maximum(cont*mask_k,mask)
        cont += 1
#     plt.imshow(mask)
#     plt.colorbar()
#     plt.show()
    labels = np.array([key for key in region.keys()])
    
    return mask,labels

def get_region_mask(video,pre_land):
    
    ## ADD EXTRA LANDMARKS ##
    
    # boosted landmarks
    boosted_landmarks = np.array(pre_land).mean(-1)

    #add forehead
    face_length = np.max(boosted_landmarks[:,1])-np.min(boosted_landmarks[:,1])
    idx_forehead = np.arange(18,27)

    aux_fh = np.array(boosted_landmarks[idx_forehead,:])
    aux_fh[:,1] = aux_fh[:,1] - face_length*0.25

    new_points = np.zeros([6,2])

    #between eyebrows
    new_points[0,:] = (boosted_landmarks[21,:] + boosted_landmarks[22,:])/2
    new_points[1,:] = (aux_fh[3,:] + aux_fh[4,:])/2

    #middle left cheek
    new_points[2,:] = (boosted_landmarks[48,:] + boosted_landmarks[3,:])/2
    new_points[3,:] = (boosted_landmarks[31,:] + boosted_landmarks[1,:])/2

    #middle left cheek
    new_points[4,:] = (boosted_landmarks[54,:] + boosted_landmarks[13,:])/2
    new_points[5,:] = (boosted_landmarks[35,:] + boosted_landmarks[15,:])/2

    #concatenate ALL
    boosted_landmarks = np.concatenate([boosted_landmarks,aux_fh,new_points],axis = 0)
    
    #get masks
    video_mean = np.mean(video[:,:,:],axis = 2).squeeze()
    masks, regions_keys = get_region_values(video_mean, boosted_landmarks[:,:])
    
    return masks, regions_keys, boosted_landmarks

def extract_video_and_landmarks(file_root, dlib_path, plot=False):
    
    def shape_to_np(shape):
        shape = list(shape.parts())
        numpy_shape=np.zeros([len(shape),2])
        for i,s in enumerate(shape):
            numpy_shape[i,0]=s.x
            numpy_shape[i,1]=s.y
        return numpy_shape
    
    #load dlib utilities
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_path)


    # list all images
    img_list = sorted(glob.glob(file_root+'*.png'))

    #get initial image and shape
    img = (io.imread(img_list[0],as_gray=True)*255).astype('uint8')
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        shape = shape_to_np(shape)
    if plot:
        plt.imshow(img)
        for pt in range(shape.shape[0]):
            plt.scatter(shape[pt,0],shape[pt,1], label="{}".format(pt))
        plt.show()
    

    #Make video matrix
    video = np.zeros([len(img_list), *img.shape])
    for t in np.arange(video.shape[0]):
        video[t]  = (io.imread(img_list[t],as_gray=True)*255).astype('uint8')
    # detect landmarks
    landmarks = np.zeros([video.shape[0],*shape.shape])
    for t in np.arange(video.shape[0]):
        img = (video[t]).astype('uint8')
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            shape = shape_to_np(shape)
        landmarks[t]=shape

    #flip array order
    landmarks = np.transpose(landmarks,(1,2,0))
    video = np.transpose(video,(1,2,0))

    return {"image_block":video, "landmarks":landmarks}

def get_grid_signals(video,boosted_landmarks,masks,grid_size=5):
    from scipy import stats
    
    ## BOUNDING BOX ##
    bb_lands=np.zeros([4,2])
    median_land = np.median(boosted_landmarks,axis=0)
    min_x = boosted_landmarks[:,0].min() - 15
    max_x = boosted_landmarks[:,0].max() + 15
    min_y = boosted_landmarks[:,1].min() - 15
    max_y = boosted_landmarks[:,1].max() + 15
    bb_lands[0,:]=[min_x,min_y]
    bb_lands[1,:]=[min_x,max_y]
    bb_lands[2,:]=[max_x,min_y]
    bb_lands[3,:]=[max_x,max_y]

    #video box and mask box
    video_box = np.array(video)
    video_box = video_box[int(min_y):int(max_y),int(min_x):int(max_x),:]
    masks_box = masks[int(min_y):int(max_y),int(min_x):int(max_x)]
    
    #turn mask into boolean 0,1 where 1 is the region of interest, 
    #in this case all labels != 0 (full face area)
    chosen_mask = np.array(masks_box)
    chosen_mask[chosen_mask>0] = 1 
    
    pct=0.2 #minimum percentage of the grid that must be included in the mask to consider it
    mask_grid = np.zeros([video_box.shape[0],video_box.shape[1]])
    grid_count=np.arange(min_x+1,max_x+1,grid_size).size*np.arange(min_y+1,max_y+1,grid_size).size
    
    #output, labels_per_grid (features x times)
    video_grid=np.zeros([grid_count,video_box.shape[-1]])
    label_grid=np.zeros([grid_count])
    
    #grid_good = np.zeros([grid_count]).astype('bool')
    success_count=1
    final_mask_grid=np.zeros(mask_grid.shape)
    for i in np.arange(video_box.shape[1],step=grid_size):
        for j in np.arange(video_box.shape[0],step=grid_size):
            polygon=[(int(i),int(j)),
                    (int(i),int(j+grid_size)),
                    (int(i+grid_size),int(j+grid_size)),
                    (int(i+grid_size),int(j)),
                    ]
            imgmask = Image.new('L', (video_box.shape[1],video_box.shape[0]), 0)
            ImageDraw.Draw(imgmask).polygon(polygon, outline=1, fill=1)
            mask_grid_ij = np.array(imgmask)*chosen_mask
            
            if mask_grid_ij.sum()>=(pct*grid_size**2):
                final_mask_grid[np.where(mask_grid_ij>0)]=success_count
                video_grid[success_count-1,:]=(video_box*mask_grid_ij[:,:,np.newaxis]).mean((0,1))
                val = stats.mode(masks_box[np.where(mask_grid_ij>0)])
    #            print(val)
                val = int(val[0])
    #            print(val)
                label_grid[success_count-1] = val
                success_count+=1
                mask_grid += mask_grid_ij
                
    label_grid = label_grid[:success_count]
    video_grid = video_grid[:success_count,:]
    
    return label_grid,video_grid,video_box,masks_box,mask_grid