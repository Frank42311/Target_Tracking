import cv2
import numpy as np
import time
import math
import pydicom
from PIL import Image
import os

# Gaussian weights used in target and candidate model.
def Gaussian_weights_in_target_model(_roi):

    '''
    :param _roi: half length and half width of region of interest (x,y).
                ps. (squered region are used in this algorithm)
    :return: gaussian weights
    '''
    '''
        The difference between two gaussian weights.
        This one uses dependent variable of the gaussian distribution, f(x).
    '''

    # Initialization
    _gaussian_weights = np.zeros([_roi[0] * 2, _roi[1] * 2])

    # Sigma for gaussian weights.
    # ps. Sigma doesn't really need to be changed, thus, it can be considered as a constant.
    _sigma = 1

    # X and Y coordinate ranges of region of interest.
    _x_range = np.linspace(-2 * _sigma, 2 * _sigma, _roi[0] * 2)
    _y_range = np.linspace(-2 * _sigma, 2 * _sigma, _roi[1] * 2)

    # Generate gaussian weights.
    for x, _x_value in enumerate(_x_range):
        for y, _y_value in enumerate(_y_range):
            # Generate gaussian values, f(x,y), and fill it in gaussian-weights matrix.
            _gaussian_weights[x][y] = math.exp(
                -(_x_value * _x_value + _y_value * _y_value) / (2 * _sigma * _sigma)) / (
                                             2 * math.pi * _sigma * _sigma)

    return _gaussian_weights

# Gaussian weights used in target tracking.
def Gaussian_weights_in_target_tracking(_roi):

    '''
    :param _roi: half length and half width of region of interest (x,y).
                ps. (squered region are used in this algorithm)
    :return: gaussian weights
    '''
    '''
        The difference between two gaussian weights.
        This one uses dependent variable of the first-order derivative of gaussian distribution, g(x) = f'(x).
            In addition, the values Take negative values, thus, values which are finally used are -g(x) here.
    '''

    # Initialization
    _gaussian_weights = np.zeros([_roi[0] * 2, _roi[1] * 2])

    # Sigma for gaussian weights.
    # ps. Sigma doesn't really need to be changed, thus, it can be considered as a constant.
    _sigma = 2

    # X and Y coordinate ranges of region of interest.
    _x_range = np.linspace(-2 * _sigma, 2 * _sigma, _roi[0] * 2)
    _y_range = np.linspace(-2 * _sigma, 2 * _sigma, _roi[1] * 2)

    # Generate gaussian weights.
    for x, _x_value in enumerate(_x_range):
        for y, _y_value in enumerate(_y_range):
            # Generate First order x partial derivative of gaussian distribution.
            _partial_derivative_x = -(x * math.exp(
                -(_x_value * _x_value + _y_value * _y_value) / (2 * _sigma * _sigma))) / (
                                             2 * math.pi * pow(_sigma, 4))
            # Generate First order y partial derivative of gaussian distribution.
            _partial_derivative_y = -(y * math.exp(
                -(_x_value * _x_value + _y_value * _y_value) / (2 * _sigma * _sigma))) / (
                                            2 * math.pi * pow(_sigma, 4))
            # Gaussian weights.
            _gaussian_weights[x][y] = _partial_derivative_x + _partial_derivative_y

    return _gaussian_weights

def Target_model(_img, _y0, _roi, _weights):

    '''
    :param _img: target image; Fixed image.
    :param _y0: center coordinates(x,y) of region of interest.
    :param _roi: half length and half width of region of interest (x,y).
                ps. (squered region are used in this algorithm)
    :param _weights: gaussian weights for convolution.
    :return: target model (an array of 128 bins)
                ps. 128 bins may be too large, this single function will cost about 1s.
                    To be more effective, bins need to be reduced.
    '''

    # X and Y coordinate ranges of region of interest.
    _x_index = range(_y0[0]-_roi[0], _y0[0]+_roi[0])
    _y_index = range(_y0[1]-_roi[1], _y0[1]+_roi[1])

    # Initialization.
    _qu = np.zeros((128,))

    # Build the target model.
    for _idx in range(len(_qu)):
        for _x_idx in range(len(_x_index)):
            for _y_idx in range(len(_y_index)):
                # ps. _rgb is actually gray scale value.
                #     The test algorithm is for RGB images.
                #     For simplicity, the function Mix_rgb() is changed and the name _rgb is remained.
                #     More details are shown in Mix_rgb()
                _rgb = Mix_rgb(_img, _x_index[_x_idx], _y_index[_y_idx])
                # Select bins, 128 bins in total.
                if _rgb == _idx*2 or _rgb == _idx*2+1:
                    _qu[_idx] += _rgb * _weights[_x_idx][_y_idx]

    # Normalization.
    _sum = np.sum(_qu)
    for _idx, _item in enumerate(_qu):
        _qu[_idx] = _qu[_idx] / _sum

    return _qu

def Candidate_model(_img, _y0, _roi, _weights):

    '''
    :param _img: target image; Fixed image.
    :param _y0: center coordinates(x,y) of region of interest.
    :param _roi: half length and half width of region of interest (x,y).
                ps. (squered region are used in this algorithm)
    :param _weights: gaussian weights for convolution.
    :return: target model (an array of 128 bins)
                ps. 128 bins may be too large, this single function will cost about 1s.
                    To be more effective, bins need to be reduced.
    '''
    '''
        ps. Totally the same as Target model.
            Only difference here is that subsequential images need to be used here instead of fixed image.
    '''

    # X and Y coordinate ranges of region of interest.
    _x_index = range(_y0[0]-_roi[0], _y0[0]+_roi[0])
    _y_index = range(_y0[1]-_roi[1], _y0[1]+_roi[1])

    # Initialization
    _pu = np.zeros((128,))

    # Build the candidate model.
    for _idx in range(len(_pu)):
        for _x_idx in range(len(_x_index)):
            for _y_idx in range(len(_y_index)):
                # ps. _rgb is actually gray scale value.
                #     The test algorithm is for RGB images.
                #     For simplicity, the function Mix_rgb() is changed and the name _rgb is remained.
                #     More details are shown in Mix_rgb()
                _rgb = Mix_rgb(_img, _x_index[_x_idx], _y_index[_y_idx])
                # Select bins, 128 bins in total.
                if _rgb == _idx*2 or _rgb == _idx*2+1:
                    _pu[_idx] += _rgb * _weights[_x_idx][_y_idx]

    # Normalization
    _sum = np.sum(_pu)
    for _idx, _item in enumerate(_pu):
        _pu[_idx] = _pu[_idx] / _sum

    return _pu

# Compare similarity in target tracking.
def Similarity_comparison(_qu, _pu):

    '''
    :param _qu: target model
    :param _pu: candidate model
    :return: similarity using Bhatta charrya Coefficient
    '''

    # Initialization
    _ratio = 0

    # Compute Bhatta charrya Coefficient
    for _idx in range(len(_qu)):
        _ratio += math.sqrt(_qu[_idx] * _pu[_idx])

    return _ratio

# Used extract intensity values of images.
def Mix_rgb(_img, _x, _y):

    '''
    :param _img: image
    :param _x: coordiante x in the image
    :param _y: coordinate y in the image
    :return: intensity information
    '''

    # This is the original test algorithm.
    '''
        This part is for rgb values.
            extract first 3 binary values in R
            extract first 3 binary values in G
            extract first 2 binary values in B
            then combine them together as a 8 binary value like gray scale 
    '''
    '''
    _r = bin(_img[_x][_y][0])[2:5]
    _g = bin(_img[_x][_y][1])[2:5]
    _b = bin(_img[_x][_y][2])[2:4]
    _mix = int(_r + _g + _b, 2)
    '''

    # gray scale value.
    _mix = _img[_x][_y]

    return _mix

# Generate n candidates for subsequential images.
def Generate_y_candidates(_img, _y0):

    '''
    :param _img: one subsequential image.
    :param _y0: center coordinates of one subsequential image.
    :return: n candidates
    '''

    # Candidate region size.
    #   ps. The center of region of interest in subsequential images doesn't move much, thus, 5 is enough.
    #       25 candidates are selected here.
    _candidate_size = 5
    _temp_value = int(_candidate_size/2)

    # Initialization.
    _y_candidates = np.zeros([_candidate_size, _candidate_size, 3])

    # X and Y coordinates of candidates in subsequential image.
    _x_index = np.linspace(_y0[0] - _temp_value, _y0[0] + _temp_value, _candidate_size, dtype=int)
    _y_index = np.linspace(_y0[1] - _temp_value, _y0[1] + _temp_value, _candidate_size, dtype=int)

    # Generate n candidates, and save x, y, intensity information into the candidate matrix.
    for x in range(len(_x_index)):
        for y in range(len(_y_index)):
            _y_candidates[x][y][0] = _x_index[x]
            _y_candidates[x][y][1] = _y_index[y]
            _y_candidates[x][y][2] = Mix_rgb(_img, _x_index[x], _y_index[y])

    return _y_candidates

# Target tracking for a single image.
def Mean_shift_target_tracking(_img, _second_img, _y0, _y_candidates, _qu, _pu, _gaussian_weights, _roi):

    '''
    :param _img: fixed image.
    :param _second_img: one subsequential image.
    :param _y0: center of one subsequential image.
    :param _y_candidates: n estimates of candidates in one subsequential image.
    :param _qu: target model.
    :param _pu: candidate model.
    :param _gaussian_weights: gaussian weights.
    :param _roi: half length and half width of region of interest (x,y).
    :return: best estimate of the candidate in one subsequential image.
    '''
    '''
        Core function of the algorithm.

        N candidates are selected in one subsequential image.
        Return the best estimation of the center of the region of interest.
    '''

    # Ratio weights.
    def Weights_in_target_tracking(_qu, _pu, _xi_rgb):

        '''
        :param _qu: target model
        :param _pu: candidate model
        :param _xi_rgb: gray scale intensity of a single pixel
        :return: ratio weights
        '''
        '''
            In target tracking function, two weights are used.
                One is gaussian weights: -g(x)
                Another one is this one, computed as follow:
                    _weights = math.sqrt(_qu[_idx] / _pu[_idx])
        '''

        # Initialization.
        _weights = 1

        # Generate ratio weights.
        for _idx in range(len(_pu)):
            if _xi_rgb == _idx * 2 or _xi_rgb == _idx * 2 + 1:
                # Avoid zero denominator.
                if _pu[_idx] == 0:
                    _weights = 1
                else:
                    _weights = math.sqrt(_qu[_idx] / _pu[_idx])

        return _weights

    # Generate best estimation of n candidates.
    def Y1_calculation(_second_img, _y_can, _qu, _pu, _gaussian_weights, _roi):

        '''
        :param _second_img: one subsequential image.
        :param _y_can: n estimates of candidates in one subsequential image.
        :param _qu: target model.
        :param _pu: candidate model.
        :param _gaussian_weights: gaussian weights.
        :param _roi: half length and half width of region of interest (x,y).
        :return: best estimate of the candidate in one subsequential image.
        '''
        '''
            Core function of the algorithm.

            N candidates are selected in one subsequential image.
            Return the best estimation of the center of the region of interest.
        '''

        # Initialization.
        # ps. y1 is calculated as follow:
        #       _y1 = _numerator / _denominator
        _numerator = 0
        _denominator = 0

        # X and Y coordinates of candidates in subsequential image.
        _x_index = range(_y_can[0] - _roi[0], _y_can[0] + _roi[0])
        _y_index = range(_y_can[1] - _roi[1], _y_can[1] + _roi[1])

        # Generate ratio weights.
        _w = Weights_in_target_tracking(_qu, _pu, 186)

        # Generate best estimation of candidates.
        for _x_idx in range(len(_x_index)):
            for _y_idx in range(len(_y_index)):
                _rgb = Mix_rgb(_second_img, _x_index[_x_idx], _y_index[_y_idx])
                _w = Weights_in_target_tracking(_qu, _pu, _rgb)
                _numerator += (_w * _rgb * _gaussian_weights[_x_idx][_y_idx])
                _denominator += (_w * _gaussian_weights[_x_idx][_y_idx])
        _y1 = _numerator / _denominator

        return _y1

    # Initialize n candidates.
    _y_candidates = Generate_y_candidates(_second_img, _y0)
    # Temp value to store estimation information.
    _y1_temp = 0

    # Compare all results of candidates.
    for _y_x_i in range(_y_candidates.shape[0]):
        for _y_y_i in range(_y_candidates.shape[1]):
            # Generate one single result of one candidate.
            _y1_result = Y1_calculation(_second_img, (int(_y_candidates[_y_x_i][_y_y_i][0]), int(_y_candidates[_y_x_i][_y_y_i][1])), _qu, _pu, _gaussian_weights, _roi)
            # Find the best candidate, which is the largest one.
            if _y1_temp <= _y1_result:
                _y1_temp = _y1_result
                _y1 = [int(_y_candidates[_y_x_i][_y_y_i][0]), int(_y_candidates[_y_x_i][_y_y_i][1])]

    return _y1

# Display results on images.
def Display_on_image(_second_img, _y1, _roi):

    '''
    :param _second_img: one subsequential image.
    :param _y1: center coordinates of one subsequential image.
    :param _roi: half length and half width of region of interest (x,y).
    :return: one subsequential image with 3 pixel wide hollow circle.
    '''

    # Need to be modified. Into circles
    _x_range = range(_y1[0] - _roi[0]-2, _y1[0] + _roi[0] + 3)
    _y_range = range(_y1[1] - _roi[1]-2, _y1[1] + _roi[1] + 3)

    # Initialization, for better view of region of interest, RGB images are used here.
    _second_img_with_dege = np.zeros([_second_img.shape[0], _second_img.shape[1], 3])

    # Convert gray scale into RGB.
    for _x in range(_second_img_with_dege.shape[0]):
        for _y in range(_second_img_with_dege.shape[1]):
            _second_img_with_dege[_x][_y][0] = _second_img[_x][_y]
            _second_img_with_dege[_x][_y][1] = _second_img[_x][_y]
            _second_img_with_dege[_x][_y][2] = _second_img[_x][_y]

    # Radius of region of interest.
    _r = _roi[0]

    # Draw a three-pixel wide hollow circle on the image
    for _x in _x_range:
        for _y in _y_range:
            if int(math.sqrt(pow(_x-_y1[0], 2) + pow(_y-_y1[1], 2))) == _r:
                _second_img_with_dege[_x][_y][0] = 0
                _second_img_with_dege[_x][_y][1] = 0
                _second_img_with_dege[_x][_y][2] = 255
            if int(math.sqrt(pow(_x-_y1[0], 2) + pow(_y-_y1[1], 2))) == _r+1:
                _second_img_with_dege[_x][_y][0] = 0
                _second_img_with_dege[_x][_y][1] = 0
                _second_img_with_dege[_x][_y][2] = 255
            if int(math.sqrt(pow(_x-_y1[0], 2) + pow(_y-_y1[1], 2))) == _r+2:
                _second_img_with_dege[_x][_y][0] = 0
                _second_img_with_dege[_x][_y][1] = 0
                _second_img_with_dege[_x][_y][2] = 255

    return _second_img_with_dege

def MR_target_tracking():

    '''
    :return: none
    '''
    '''
        Core function:
            load a
    '''

    # Region of interest information.
    # Center and area.
    y0 = (132, 105)
    roi = (23, 23)

    # Gaussian weights f(x) and -g(x)
    weights = Gaussian_weights_in_target_model(roi)
    gaussian_weights = Gaussian_weights_in_target_tracking(roi)

    # Fixed image
    first_image_path = r'Abdominal_Data\Reference\refLung_001.dcm'
    # Convert dcm to ndarray.
    first_image = pydicom.read_file(first_image_path).pixel_array

    # Initialize target model.
    qu = Target_model(first_image, y0, roi, weights)

    # Load and save path.
    frames_load_path = r"Abdominal_Data\Moving"
    frames_save_path = r"frames"

    # Process 200 images.
    count = 0
    while count<200:

        count += 1

        # Time calculating
        start = time.process_time()
        print("Processing: %d.dcm" % count)

        if count < 10:
            second_img_name = frames_load_path + "\Moving_00%d" % count + "\Moving_00%d.dcm" % count
            second_img = pydicom.read_file(second_img_name).pixel_array
        elif count < 100:
            second_img_name = frames_load_path + "\Moving_0%d" % count + "\Moving_0%d.dcm" % count
            second_img = pydicom.read_file(second_img_name).pixel_array
        else:
            second_img_name = frames_load_path + "\Moving_%d" % count + "\Moving_%d.dcm" % count
            second_img = pydicom.read_file(second_img_name).pixel_array

        # Initialize y candidates in one subsequential image.
        y_candidates = Generate_y_candidates(second_img, y0)

        # Candidate image.
        pu = Candidate_model(second_img, y0, roi, weights)
        # Compute Bhatta charrya Coefficient for better result.
        ro0 = Similarity_comparison(qu, pu)

        # Fine best candidate in one subsequential image.
        y1 = Mean_shift_target_tracking(first_image, second_img, y0, y_candidates, qu, pu, gaussian_weights, roi)

        # Generate candidate model for y1 found above.
        y1_pu = Candidate_model(second_img, y1, roi, weights)
        # Compute Bhatta charrya Coefficient.
        ro1 = Similarity_comparison(qu, y1_pu)

        # Set max iteration to avoid too many iterations.
        # ps. Max iteration is 10
        count_iteration = 0

        # Evaluate result using Bhatta charrya Coefficient.
        while ro1 < ro0:
            if count_iteration <= 10:
                print('\titeration: ', count_iteration)
                count_iteration += 1
                # Reset coordinates as follow.
                y1[0] = int((y0[0] + y1[0]) / 2)
                y1[1] = int((y0[1] + y1[1]) / 2)
                # Regenerate candidate model and Bhatta charrya Coefficient using new coordinates.
                y1_pu = Candidate_model(second_img, y1, roi, weights)
                ro1 = Similarity_comparison(qu, y1_pu)
            else:
                break

        # Display results on images.
        second_image_with_edge = Display_on_image(second_img, y1, roi)
        # Save images.
        cv2.imwrite(frames_save_path + r"\frame%d.jpg" % count, second_image_with_edge, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Time calculating.
        end = time.process_time()
        # Calculate the time needed by the enhancing algorithm
        print("\tFinish processing: %d" % count)
        print("\tTime:", end - start)

# Combine images into video, 24 frames per second.
def Frame2video(im_dir, video_dir, fps):

    im_list = os.listdir(im_dir)
    im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for i in im_list:
        im_name = os.path.join(im_dir + i)
        frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
        videoWriter.write(frame)
    videoWriter.release()
    print('finish')

if __name__ == '__main__':

    # Generate 200 frames.
    MR_target_tracking()
    # Combine images into video, 24 frames per second.
    Frame2video(r'frames\\', r'MR.avi', 24)