import sys, os, argparse
import numpy as np
from imageio import imread, imsave
import render_utils as utils
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--in_root',  default='./output_blender')
# parser.add_argument('--in_dir',   default='image_Circle_0')
# parser.add_argument('--out_dir',  default='')
parser.add_argument('--reload',   action='store_false', default=False)
parser.add_argument('--mute',     action='store_false', default=True)
args = parser.parse_args()

class FlowCalibrator:
    def __init__(self, imgs=[], bgs = []):
        self.imgs = imgs
        self.bgs = bgs
        if imgs != []:
            self.h = imgs[0].shape[0]
            self.w = imgs[0].shape[1]
            if not args.mute:
                print('Image height width %dX%d' % (self.h, self.w))
            self.mask = imgs[1]
            self.rho  = imgs[0]

    def processing(self, img, i, thres, out_name):
        histo = cv2.calcHist([img], [0], None, [256], [0,256])
        # print(histo)
        plt.plot(histo)
        t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thres = np.linspace(0, 1, 255)
        # print(thres)
        plt.plot(thres, [t]*255)
        if not os.path.exists(os.path.join(out_name, 'bitcode')):
            os.mkdir(os.path.join(out_name, 'bitcode'))

        plt.savefig(out_name+f'/bitcode/{i:02d}_histo_{t}.png')
        plt.clf()
        cv2.imwrite(out_name+f'/bitcode/{i:02d}_result.png', t_otsu)

        return t_otsu, t

    def obtainImgBinaryCode(self, sub_imgs, h, w, thres=190, out_name='.'):
        if not args.mute:
            print('Obtaining Image binary code (%dx%dx%d)' % (len(sub_imgs), h,w))
        binary_code = np.chararray((h, w), itemsize=1)
        binary_code[:] = ''
        for i, img in enumerate(sub_imgs):
            bit_code = np.chararray((h,w), itemsize=1)
            # int_code = np.ones((h,w))
            img, thres = self.processing(img, i, thres, out_name)
            # print(img.shape, type(img), thres)
            bit_code[img >  thres] = '1'
            bit_code[img <= thres] = '0'
            # int_code[img <= thres] = 0
            # print(int_code)
            # cv2.imwrite(f'./{i:02d}.png', int_code*255)
            binary_code = binary_code + bit_code
        if not args.mute:
            print(binary_code[int(h/2)-2:int(h/2)+2,int(w/2-2):int(w/2)+2], len(binary_code[0,0]))
        return binary_code

    def obtainImgBinaryCode_hyebin(self, sub_imgs, h, w):
        if not args.mute:
            print('Obtaining Image binary code (%dx%dx%d)' % (len(sub_imgs), h,w))
        binary_code = np.chararray((h, w)); binary_code[:]=''
        for i, img in enumerate(sub_imgs):
            bit_code = np.chararray((h,w), itemsize=1)
            bit_code[img >  self.bgs[i]] = '1'
            bit_code[img <= self.bgs[i]] = '0'
            binary_code = binary_code + bit_code
        if not args.mute:
            print(binary_code[:2,:2], len(binary_code[0,0]))
        return binary_code

    def obtainImgBinaryCode_farenback(self, sub_imgs, h, w):
        if not args.mute:
            print('Obtaining Image binary code (%dx%dx%d)' % (len(sub_imgs), h,w))
        flow_x = np.zeros_like(sub_imgs[0], dtype=np.float32)
        flow_y = np.zeros_like(sub_imgs[0], dtype=np.float32)
        for i, img in enumerate(sub_imgs):
            flow = cv2.calcOpticalFlowFarneback(self.bgs[i], img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_x += flow[...,0]
            flow_y += flow[...,1]
        flow_x[self.mask >= 200] = 0
        flow_y[self.mask >= 200] = 0

        return flow_x, flow_y

    def findCorrespondence(self, out_name):
        self.flow_x_idx = np.zeros((self.h, self.w), dtype=np.int64) # np.int64
        self.flow_y_idx = np.zeros((self.h, self.w), dtype=np.int64) # np.int64
        self.x_grid = np.tile(np.linspace(0, self.w-1, self.w), (self.h, 1)).astype(np.int64) # int
        self.y_grid = np.tile(np.linspace(0, self.h-1, self.h), (self.w, 1)).T.astype(np.int64) # int
        if not args.mute:
            print('Finding correspondence with graycode pattern')
        self.img_code  = self.obtainImgBinaryCode(self.imgs[2:], self.h, self.w, 180, out_name)
        # self.img_code  = self.obtainImgBinaryCode_hyebin(self.imgs[2:], self.h, self.w)
        # self.flow_x_idx, self.flow_y_idx  = self.obtainImgBinaryCode_farenback(self.imgs[2:], self.h, self.w)
        # self.saveFlow(self.flow_x_idx, self.flow_y_idx, self.rho, out_name)
        self.findCorrespondenceGraycode(os.path.join(out_name, 'flow'))

    def findCorrespondenceGraycode(self, out_name):
        digit_code = [int(code, 2) for code in self.img_code.flatten()]
        digit_code = np.array(digit_code).reshape(self.h, self.w)
        self.flow_x_idx = np.mod(digit_code, self.w)# .astype(np.float32) # int(self.w/2))
        self.flow_y_idx = np.divide(digit_code, self.w)# .astype(np.float32) # int(self.w/2))

        self.flow_x_idx -= self.x_grid
        self.flow_y_idx -= self.y_grid
        # self.flow_x_idx[self.mask >= 200] = 0
        # self.flow_y_idx[self.mask >= 200] = 0
        # self.flow_x_idx[self.mask < 55] = 0
        # self.flow_y_idx[self.mask < 55] = 0
        self.saveFlow(self.flow_x_idx, self.flow_y_idx, self.mask, out_name)

    def flowWithRho(self, flow_color, rho):
        h = rho.shape[0]
        w = rho.shape[1]
        # flow_rho = flow_color * np.ones((h,w,3)).astype(float)
        flow_rho = flow_color * np.tile(rho.reshape(h,w,1), (1,1,3)).astype(float) /255
        return flow_rho.astype(np.uint8)

    def writeFlowBinary(self, flow, filename):
        flow = flow.astype(np.float32)
        with open(filename, 'wb') as f:
            magic = np.array([202021.25], dtype=np.float32) 
            h_w   = np.array([flow.shape[0], flow.shape[1]], dtype=np.int32)
            magic.tofile(f)
            h_w.tofile(f)
            flow.tofile(f)

    def remove_grid(self, flow, window_size, mask):
        half = window_size //2
        new_flow = np.zeros_like(flow)
        check_flow = np.zeros((flow.shape[0], flow.shape[1], 3))

        for y in range(half, flow.shape[0] - half):
            for x in range(half, flow.shape[1] - half):
                window = flow[y-half:y+half+1, x-half:x+half+1]
                mean_flow = np.mean(window, axis=(0,1))
                if mask[y,x] != 0:
                    if abs(flow[y,x][0] - mean_flow[0]) > 30:
                        new_flow[y,x][0] = mean_flow[0]
                        check_flow[y,x] = (255,255,0)
                    else:
                        new_flow[y,x][0] = flow[y,x][0]
                    
                    if abs(flow[y,x][1] - mean_flow[1]) > 30:
                        new_flow[y,x][1] = mean_flow[1]
                        check_flow[y,x] = (255,0,255)
                    else:
                        new_flow[y,x][1] = flow[y,x][1]
                    # print(abs(flow[y,x][0] - mean_flow[0]), abs(flow[y,x][1] - mean_flow[1]))
                        # new_flow[y,x] = mean_flow
                else:
                    new_flow[y,x] = flow[y,x]

        return new_flow, check_flow

    def remove_grid_2(self, flow, mask):
        half = 1
        new_flow = np.zeros_like(flow)
        check_flow = np.zeros((flow.shape[0], flow.shape[1], 3))

        for y in range(half, flow.shape[0] - half):
            for x in range(half, flow.shape[1] - half):
                if mask[y,x] != 0:
                    center = flow[y,x]
                    hor_diff1 = flow[y-1,x] - flow[y+1,x]
                    hor_diff2 = flow[y-1,x] - center
                    ver_diff1 = flow[y,x-1] - flow[y,x+1] 
                    ver_diff2 = flow[y,x-1] - center

                    if abs(hor_diff1[0]**2 + hor_diff1[1]**2) < 20 and abs(hor_diff2[0]**2 + hor_diff2[1]**2) > 50:
                        check_flow[y,x] = (255,255,0)
                        new_flow[y,x] = (flow[y-1,x] + flow[y+1,x])/2
                    if abs(ver_diff1[0]**2 + ver_diff1[1]**2) < 20 and abs(ver_diff2[0]**2 + ver_diff2[1]**2) > 50:
                        check_flow[y,x] = (255,255,255)
                        new_flow[y,x] = (flow[y,x-1] + flow[y,x+1])/2

                    else: 
                        new_flow[y,x] = flow[y,x]
                else:
                    new_flow[y,x] = flow[y,x]

        return new_flow, check_flow


    def saveFlow(self, flow_x, flow_y, rho, out_name):
        h = flow_x.shape[0]
        w = flow_x.shape[1]
        flow = np.zeros((h,w,2))
        flow[:,:,1] = flow_x; flow[:,:,0] = flow_y

        # masking flow
        # print(np.max(self.mask/255))
        if not args.mute:
            print(out_name)
        flow *= np.tile(self.mask.reshape(h,w,1), (1,1,2))/255
        flow_color = utils.flowToColor(flow)
        # imsave(os.path.join(out_name+'2.png'), flow_color.astype(np.uint8))
        imsave(os.path.join(out_name+ '.png'), self.flowWithRho(flow_color, rho))  # rho

        # for testing
        rho_1 = np.ones(rho.shape)*255 - self.rho
        # imsave(os.path.join(out_name+ '3.png'), self.flowWithRho(flow_color, rho_1))  # rho
        
        new_flow, check_flow = self.remove_grid_2(flow, rho_1)
        flow_color = utils.flowToColor(new_flow)
        imsave(os.path.join(out_name+ '4.png'), flow_color.astype(np.uint8))  # rho
        # cv2.imwrite(os.path.join(out_name+ '4_check.png'), check_flow)
        
        utils.writeFlowBinary(new_flow, out_name + '4.flo')
        utils.writeFlowBinary(flow, out_name + '.flo')

def readImgOrLoadNpy(folder):
    if args.reload and os.path.exists(folder):# os.path.join(args.in_root, 'imgs.npy')):

        # not used
        if not args.mute:
            print('Loading imgs.npy')
        imgs = np.load(os.path.join(args.in_root, 'imgs.npy'))

    else:
        img_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
        
        gray_folder = os.path.join(folder, 'gray')
        gray_list = [os.path.join(gray_folder, f) for f in os.listdir(gray_folder) if f.endswith('.png')]
        # os.path.join(folder, f) for f in os.listdir(folder) if os.isdir(folder)]
        
        # print(gray_folder)
        # print(img_list)
        # print(gray_list)

        final = []
        # make the attenuation map
        # add the mask
        for img in img_list:
            # make alpha
            if '0_alpha_0.png' in img:
                image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                # name = img.split('/')[-1]
                # name = name.split('.')[0]
                alpha = image[:,:,3]
                alpha_name = img[:-4]
                cv2.imwrite(alpha_name+'_alpha.png', alpha)
                print(alpha_name+'_alpha.png')

            # make mask
            elif '0_mask_2.png' in img:
                image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                # name = img.split('/')[-1]
                # name = name.split('.')[0]
                mask = image[:,:,3]
                mask_name = img[:-4]
                cv2.imwrite(mask_name+'_mask.png', mask)
                print(mask_name+'_mask.png')

            # make mask
            elif 'att_2.png' in img:
                att_path = img
                image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray2 = image[:,:,:3].mean(axis=2)
                # name = img.split('/')[-1]
                # name = name.split('.')[0]
                att = image[:,:,3]

        final.append(alpha_name+'_alpha.png')
        final.append(mask_name+'_mask.png')


        '''cv2.imwrite(att_path[:-4]+'_gray1.png', gray1)
        print(att_path[:-4]+'_gray1.png')
        cv2.imwrite(att_path[:-4]+'_gray2.png', gray2)
        print(att_path[:-4]+'_gray2.png')


        print(gray1.shape, gray2.shape)
        print(np.max(gray1), np.max(gray2))
        print(np.max(mask), np.max(alpha), np.max(mask/255))
        gray1 = np.array(gray1, dtype='float64')
        print(np.max(gray1*mask/255), np.max(gray2*mask/255))
        
        cv2.imwrite(att_path[:-4]+'_mask_1.png', gray1*mask/255)
        print(att_path[:-4]+'_mask_1.png')
        # final.append(att_path[:-4]+'_gray.png')


        cv2.imwrite(att_path[:-4]+'_alpha_1.png', gray1*alpha/255)
        print(att_path[:-4]+'_alpha_1.png')
        # final.append(att_path[:-4]+'_gray.png')

        cv2.imwrite(att_path[:-4]+'_mask_2.png', gray2*mask/255)
        print(att_path[:-4]+'_mask_2.png')
        # final.append(att_path[:-4]+'_mask_2.png')

        cv2.imwrite(att_path[:-4]+'_alpha_2.png', gray2*alpha/255)
        print(att_path[:-4]+'_alpha_2.png')
        # final.append(att_path[:-4]+'_gray.png')

        cv2.imwrite(att_path[:-4]+'_mask_2.png', gray2*mask/255)
        print(att_path[:-4]+'_mask_2.png')
        # final.append(att_path[:-4]+'_mask_2.png')

        cv2.imwrite(att_path[:-4]+'_alpha_2.png', gray2*alpha/255)
        print(att_path[:-4]+'_alpha_2.png')
        # final.append(att_path[:-4]+'_gray.png')'''





        bg = []
        gray_list.sort()
        for img in gray_list:
            if 'bg' in img:
                bg.append(img)
            else:
                print(img)
                final.append(img)
        # print(bg)
        # print(final)
        if not args.mute:
            print('Totaling %d images' % len(final))
            
        imgs = utils.readImgFromList(final)
        utils.listRgb2Gray(imgs)
        bgs = []
        # np.save(os.path.join(args.in_root, 'imgs.npy'), imgs)
        # bgs = utils.readImgFromList(bg)
        # utils.listRgb2Gray(bgs)

    return imgs, bgs

def checkImgNumber(imgs):
    img_num = len(imgs)
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    if not (img_num == 2 + int(np.log2(h)) + int(np.log2(w))):
        raise Exception('Not correct image number: %dX%dX%d' % (img_num, h, w))

if __name__ == '__main__':
    folders = [os.path.join(args.in_root, f) for f in os.listdir(args.in_root)]

    # object folders
    for folder in folders:
        if "previous" not in folder:
            folds = [os.path.join(folder, f) for f in os.listdir(folder) if not f.endswith('.png') or not f.endswith('.txt') or not f.endswith('.flo')]
            print(folder, '\n', folds)

            # background folders
            # for fold in folds:
            imgs, bgs = readImgOrLoadNpy(folder)
            checkImgNumber(imgs)

            out_dir = os.path.join(folder, 'flow')
            print(out_dir)
            calibrator = FlowCalibrator(imgs, bgs)
            calibrator.findCorrespondence(folder)

