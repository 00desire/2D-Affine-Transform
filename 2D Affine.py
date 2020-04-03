import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import itertools

def findMatches(img_stream,kp_master, dc_master):

    #Match the point with the original image back
    #Compute ORB KeyPoints and Descriptor in input image 
    orb.setMaxFeatures(len(kp_master))
    kp_inp, dc_inp = orb.detectAndCompute(img_stream, None)
   
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(dc_inp, dc_master, None)

    matches.sort(key=lambda x: x.distance, reverse=False)
    
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    return matches,kp_inp,kp_master

def findCnts(mask):
    thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = np.asarray(imutils.grab_contours(cnts))
    cnt_raveled =[]
    for cnt in cnts:
        cnt_raveled.append(cnt.reshape(cnt.shape[0],cnt.shape[2]))
    return cnt_raveled

def getFilteredKeypoints(image,mask):
    
    kp_filtered = []
    dc_filtered = np.zeros([1,32],dtype=np.uint8)
    
    #Compute ORB KeyPoints and Descriptor in mask area 
    kp_master, dc_master = orb.detectAndCompute(image, None) 
    
    cnts = findCnts(mask)
    
    for cnt in cnts:
        passed = [cv2.pointPolygonTest(cnt,kp.pt,True) >= 0 for kp in kp_master]

        kp_passed = list(itertools.compress(kp_master, passed))
        kp_filtered.extend(kp_passed)
        
        dc_passed = dc_master[passed,:]
        dc_filtered = np.append(dc_filtered,dc_passed, axis=0)
        
    ## Remove first item of dc_filtered
    dc_filtered = dc_filtered[1:dc_filtered.shape[0],:]
    
    if (figures):
        img2 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.drawKeypoints(image,kp_filtered,img2,color=(0,255,0), flags=0)
        plt.figure(figsize=(15,30))
        plt.title('Filtered Ratio: {}/{} = {}'.format(len(kp_filtered),len(kp_master),(len(kp_filtered)/len(kp_master))))
        plt.imshow(img2,cmap='gray')
        plt.show()
        
    return kp_filtered, dc_filtered
    
if __name__ == "__main__": 
    ## Create ORB
    n_features = 10000
    orb = cv2.ORB_create(n_features)

    ## Read images
    img_ref = cv2.imread('reference.png')
    img_ref_mask = cv2.imread('reference_mask.png')

    img_to_align = cv2.imread('image_to_align.png')
    
    ## Get reference keypoints
    kp_ref, dc_ref = getFilteredKeypoints(img_ref,img_ref_mask)

    ## Find matches of img_to_align to ref image
    m,kp_inp,kp_ref = findMatches(img_to_align,kp_ref, dc_ref)

    ## Get matched points
    pt_inp = np.array([kp_inp[x.queryIdx].pt for x in m])
    pt_ref = np.array([kp_ref[x.trainIdx].pt for x in m])

    ## Get affine transform matrix
    M,_ = cv2.estimateAffinePartial2D(np.float32(pt_inp),np.float32(pt_ref),_,cv2.RANSAC,3,2000,0.99,30)

    ## 2D(4DOF) Affine Transformation 
    img_aligned = cv2.warpAffine(img_to_align,M,img_ref.shape[1],img_ref.shape[0])

    ## Plot
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(img_ref)
    plt.subplot(1,2,2)
    plt.imshow(img_aligned)
      
