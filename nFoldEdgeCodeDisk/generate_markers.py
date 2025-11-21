import cv2
import numpy as np

from generate_marker_codes import getRingCodes

def create_marker(size, cx, cy, r_inner_data, r_outer_data, bits, bits_list,
                  r_inner_n_fold, r_outer_n_fold, folds):
    # Prepare canvas and geometry
    img = np.ones((size, size), dtype=np.uint8) * 255  # start white
    
    # Create coordinate grids (centered, with y positive upwards)
    ys = np.arange(size)
    xs = np.arange(size)
    xv, yv = np.meshgrid(xs, ys)
    x = xv - cx
    y = cy - yv  # invert y so positive is up
    radii = np.sqrt(x.astype(np.float32) ** 2 + y.astype(np.float32) ** 2)
    angles = np.arctan2(y, x)
    angles = (angles + 2 * np.pi) % (2 * np.pi)  # map to [0, 2π)

    # Fill data slices
    slice_angle = 2 * np.pi / bits
    for i in range(bits):
        start = i * slice_angle
        end = start + slice_angle
        mask = (angles >= start) & (angles < end) & (radii <= r_outer_data) & (radii >= r_inner_data)
        color = 255*bits_list[i]
        #print(bits_list[i])
        img[mask] = color

    # Fill N-fold slices
    slice_angle = 2 * np.pi / folds
    for i in range(folds):
        start = i * slice_angle
        end = start + slice_angle
        mask = (angles >= start) & (angles < end) & (radii <= r_outer_n_fold) & (radii >= r_inner_n_fold)
        color = 255 if i%2==0 else 0
        img[mask] = color

    return img


bits = 8
transitions = 2
print("Generating ring codes for bits:", bits, "transitions:", transitions)
ring_codes = getRingCodes(bits, transitions)
print("Generated ring codes:", ring_codes)

size = 750
cx, cy = size // 2, size // 2

# ===== Dataring =====
r_outer_data = size//2
r_inner_data = int(r_outer_data * 2/3)

# ===== N-fold edge marker =====
r_outer_n_fold = int(r_outer_data * 1/2)
r_inner_n_fold = 0
folds = 10


list_of_img = []
for i in range(len(ring_codes)):
    # ===== code =====
    code = ring_codes.pop()
    bits_list = [int(x) for x in format(code & ((1 << bits) - 1), '0{}b'.format(bits))]
    print(bits_list)


    img = create_marker(size, cx, cy, r_inner_data, r_outer_data, bits, bits_list,
                        r_inner_n_fold, r_outer_n_fold, folds)
    list_of_img.append(img)

    # save individual marker
    cv2.imwrite(f"nFoldEdgeCodeDisk/8BitMarkers/generated_marker_{code}.png", img)

# put all markers together in one 5x5 grid where there is free space between each marker 
img = np.ones((size*7, size*7), dtype=np.uint8) * 255  # start white
for i, marker_img in enumerate(list_of_img):
    row = i // 3 + (i // 3) % 3 + 1
    col = i % 3 + (i % 3) % 3 + 1
    img[row*size:(row+1)*size, col*size:(col+1)*size] = marker_img

cv2.imshow("test_img", img)
cv2.waitKey(0)

cv2.imwrite("generated_marker.png", img)



