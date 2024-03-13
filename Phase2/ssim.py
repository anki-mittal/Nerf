import cv2
import numpy as np
import os
import glob
import re

def cv_ssim(img1, img2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    """
    if img1.dtype != np.uint8 or img2.dtype != np.uint8:
        raise ValueError("Images must be 8-bit")
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same size")
    
    k1, k2 = 0.01, 0.03
    L = 255
    C1, C2 = (k1*L)**2, (k2*L)**2

    I1, I2 = img1.astype(np.float64), img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

    sigma1_2 = cv2.GaussianBlur(I1**2, (11, 11), 1.5) - mu1**2
    sigma2_2 = cv2.GaussianBlur(I2**2, (11, 11), 1.5) - mu2**2

    sigma12 = cv2.GaussianBlur(I1*I2, (11, 11), 1.5) - mu1*mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_2 + sigma2_2 + C2))
    val = ssim_map.mean()

    return val

folder_path1 = '/home/ankit/Documents/STUDY/RBE594/p2/amittal_p2/Phase2/structure-from-motion/rkulkarni1_p2/Phase2/data/ship/test'
folder_path2 = '/home/ankit/Documents/STUDY/RBE594/p2/amittal_p2/Phase2/structure-from-motion/rkulkarni1_p2/Phase2/outputs/novel_views_ship'
pattern = re.compile(r'r_\d+\.png$')
# Sort the list of images to ensure they are compared in order
images1 = sorted([img for img in glob.glob(os.path.join(folder_path1, 'r_*.png')) if pattern.match(os.path.basename(img)) and 'depth' not in img])

images2 = sorted(glob.glob(os.path.join(folder_path2, '*.png')))
ssimloss = []
print(len(images1))
if len(images1) != len(images2):
    print("The folders contain a different number of images. Please ensure they are the same.")
else:
    for img_path1, img_path2 in zip(images1, images2):
        # img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)


        image_with_alpha = cv2.imread(img_path1, cv2.IMREAD_UNCHANGED)

        # Check if the image has an alpha channel
        if image_with_alpha.shape[2] == 4:
            # Split the image into its channels
            b, g, r, a = cv2.split(image_with_alpha)
            
            # Create a white background image
            white_background = np.ones_like(a) * 255
            
            # Use the alpha channel as a mask to blend the transparent areas with the white background
            b = b * (a / 255.0) + white_background * (1 - a / 255.0)
            g = g * (a / 255.0) + white_background * (1 - a / 255.0)
            r = r * (a / 255.0) + white_background * (1 - a / 255.0)

            # Merge the channels back together
            blended = cv2.merge([b, g, r])
            
            # Convert the blended image to grayscale
            gray_image = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            # If there's no alpha channel, just convert the image to grayscale
            gray_image = cv2.cvtColor(image_with_alpha, cv2.COLOR_BGR2GRAY)

        img1 = gray_image
        
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print(f"Error loading images {img_path1} or {img_path2}")
            continue
        
        # Calculate SSIM
        img1 = cv2.resize(img1, (389, 389))
        ssim_value = cv_ssim(img1, img2)
        ssimloss.append(ssim_value)
    
    print(f"SSIM between", sum(ssimloss)/len(ssimloss))

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# def cv_ssim(img1, img2):
#     """
#     Calculate the Structural Similarity Index (SSIM) between two images.
#     """
#     if img1.dtype != np.uint8 or img2.dtype != np.uint8:
#         raise ValueError("Images must be 8-bit")
#     if img1.shape != img2.shape:
#         raise ValueError("Images must have the same size")
    
#     k1, k2 = 0.01, 0.03
#     L = 255
#     C1, C2 = (k1*L)**2, (k2*L)**2

#     I1, I2 = img1.astype(np.float64), img2.astype(np.float64)

#     mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
#     mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)

#     sigma1_2 = cv2.GaussianBlur(I1**2, (11, 11), 1.5) - mu1**2
#     sigma2_2 = cv2.GaussianBlur(I2**2, (11, 11), 1.5) - mu2**2

#     sigma12 = cv2.GaussianBlur(I1*I2, (11, 11), 1.5) - mu1*mu2

#     ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
#                ((mu1**2 + mu2**2 + C1) * (sigma1_2 + sigma2_2 + C2))
#     val = ssim_map.mean()

#     return val, ssim_map

# # Read two images
# image_path1 = '/home/ankit/Documents/STUDY/RBE594/p2/amittal_p2/Phase2/structure-from-motion/rkulkarni1_p2/Phase2/outputs/novel_views_lego/img_0.png'
# image_path2 = '/home/ankit/Documents/STUDY/RBE594/p2/amittal_p2/Phase2/structure-from-motion/rkulkarni1_p2/Phase2/data/lego/test/r_0.png'
# # img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)

# img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
# # Read the image with the alpha channel
# image_with_alpha = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)

# # Check if the image has an alpha channel
# if image_with_alpha.shape[2] == 4:
#     # Split the image into its channels
#     b, g, r, a = cv2.split(image_with_alpha)
    
#     # Create a white background image
#     white_background = np.ones_like(a) * 255
    
#     # Use the alpha channel as a mask to blend the transparent areas with the white background
#     b = b * (a / 255.0) + white_background * (1 - a / 255.0)
#     g = g * (a / 255.0) + white_background * (1 - a / 255.0)
#     r = r * (a / 255.0) + white_background * (1 - a / 255.0)

#     # Merge the channels back together
#     blended = cv2.merge([b, g, r])
    
#     # Convert the blended image to grayscale
#     gray_image = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2GRAY)
# else:
#     # If there's no alpha channel, just convert the image to grayscale
#     gray_image = cv2.cvtColor(image_with_alpha, cv2.COLOR_BGR2GRAY)

# img2 = gray_image

# # Ensure the images are not None (i.e., check if they're loaded correctly)
# if img1 is None or img2 is None:
#     print("Error loading images.")
# else:
#     # Calculate SSIM
#     img1 = cv2.resize(img1, (800, 800))
#     imgplot = plt.imshow(img1)
#     plt.show()
#     ssim_value, ssim_map = cv_ssim(img1, img2)
#     print(f"SSIM: {ssim_value}")